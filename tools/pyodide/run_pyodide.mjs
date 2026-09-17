#!/usr/bin/env node

import { readFileSync, writeFileSync } from "node:fs";
import { basename } from "node:path";
import { pathToFileURL } from "node:url";

const USAGE = `Usage: run_pyodide.mjs --pyodide-module PATH --wheel PATH --docopt-wheel PATH --script PATH [--sample-data-dir PATH] [--output PATH] -- [script args...]`;
const PYODIDE_PACKAGES = [
  "certifi",
  "charset-normalizer",
  "click",
  "contourpy",
  "cycler",
  "decorator",
  "fonttools",
  "fsspec",
  "h5py",
  "idna",
  "jinja2",
  "joblib",
  "kiwisolver",
  "lazy-loader",
  "markupsafe",
  "matplotlib",
  "mpmath",
  "narwhals",
  "numpy",
  "packaging",
  "pandas",
  "pillow",
  "platformdirs",
  "pyparsing",
  "python-dateutil",
  "pytz",
  "pyyaml",
  "requests",
  "scikit-learn",
  "scipy",
  "six",
  "sympy",
  "threadpoolctl",
  "tqdm",
  "typing-extensions",
  "tzdata",
  "urllib3",
  "wrapt",
];
const MICROPIP_CONSTRAINTS = ["mne==1.10.0", "sympy==1.14.0", "threadpoolctl==3.6.0"];

function fail(message) {
  console.error(`${message}\n${USAGE}`);
  process.exitCode = 2;
}

function parseArguments(argv) {
  const separator = argv.indexOf("--");
  const options = separator === -1 ? argv : argv.slice(0, separator);
  const scriptArgs = separator === -1 ? [] : argv.slice(separator + 1);
  const result = { scriptArgs };
  for (let index = 0; index < options.length; index += 1) {
    const option = options[index];
    if (!["--pyodide-module", "--wheel", "--docopt-wheel", "--script", "--sample-data-dir", "--output"].includes(option)) {
      throw new Error(`Unknown option: ${option}`);
    }
    const value = options[index + 1];
    if (!value || value.startsWith("--")) {
      throw new Error(`Missing value for ${option}`);
    }
    result[option.slice(2).replaceAll("-", "_")] = value;
    index += 1;
  }
  for (const required of ["pyodide_module", "wheel", "docopt_wheel", "script"]) {
    if (!result[required]) {
      throw new Error(`Missing required option --${required.replaceAll("_", "-")}`);
    }
  }
  return result;
}

function copyToPyodide(pyodide, hostPath, guestPath) {
  pyodide.FS.writeFile(guestPath, readFileSync(hostPath));
}

async function main() {
  let args;
  try {
    args = parseArguments(process.argv.slice(2));
  } catch (error) {
    fail(error.message);
    return;
  }

  const pyodideModule = await import(pathToFileURL(args.pyodide_module).href);
  const pyodide = await pyodideModule.loadPyodide();
  const stdout = [];
  const stderr = [];
  pyodide.setStdout({
    batched: (message) => {
      stdout.push(message);
      process.stdout.write(message);
    },
  });
  pyodide.setStderr({
    batched: (message) => {
      stderr.push(message);
      process.stderr.write(message);
    },
  });

  await pyodide.loadPackage(["micropip", ...PYODIDE_PACKAGES]);

  const scriptPath = "/tmp/eegprep-script.py";
  copyToPyodide(pyodide, args.script, scriptPath);

  if (args.sample_data_dir) {
    const sampleData = ["eeglab_data.set", "eeglab_data.fdt"];
    pyodide.FS.mkdirTree("/tmp/eegprep-sample-data");
    for (const name of sampleData) {
      copyToPyodide(pyodide, `${args.sample_data_dir}/${name}`, `/tmp/eegprep-sample-data/${name}`);
    }
  }

  // Keep the original PEP 427 names: micropip uses the filename to identify
  // a local wheel. ``emfs:`` makes the transport work through Pyodide's
  // virtual filesystem in both the Node harness and a browser worker.
  const docoptWheelPath = `/tmp/${basename(args.docopt_wheel)}`;
  const eegprepWheelPath = `/tmp/${basename(args.wheel)}`;
  copyToPyodide(pyodide, args.docopt_wheel, docoptWheelPath);
  copyToPyodide(pyodide, args.wheel, eegprepWheelPath);

  const installCode = `
import micropip
await micropip.install("threadpoolctl==3.6.0", reinstall=True)
await micropip.install("sympy==1.14.0", reinstall=True)
await micropip.install(${JSON.stringify(`emfs:${docoptWheelPath}`)})
await micropip.install(
    ${JSON.stringify(`emfs:${eegprepWheelPath}`)},
    constraints=${JSON.stringify(MICROPIP_CONSTRAINTS)},
)
`;
  await pyodide.runPythonAsync(installCode);

  // Package loading and installation are streamed for diagnostics, but a
  // requested output file must contain only the target script's JSON/text.
  stdout.length = 0;
  stderr.length = 0;

  const scriptCode = `
import os
import runpy
import sys
${args.sample_data_dir ? 'os.environ["EEGPREP_SAMPLE_DATA"] = "/tmp/eegprep-sample-data"' : ""}
sys.argv = ${JSON.stringify([scriptPath, ...args.scriptArgs])}
try:
    runpy.run_path(${JSON.stringify(scriptPath)}, run_name="__main__")
except SystemExit as exc:
    if exc.code not in (None, 0):
        raise
`;
  try {
    await pyodide.runPythonAsync(scriptCode);
  } finally {
    if (args.output) {
      writeFileSync(args.output, stdout.join(""));
    }
  }
}

main().catch((error) => {
  console.error(error?.stack ?? error);
  process.exitCode = 1;
});
