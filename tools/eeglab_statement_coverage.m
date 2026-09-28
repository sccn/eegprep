function eeglab_statement_coverage(config_file)
% Native TestRunner statement coverage, with a fixed, independently frozen scope.
import matlab.unittest.TestRunner;
import matlab.unittest.plugins.CodeCoveragePlugin;
import matlab.unittest.plugins.XMLPlugin;
import matlab.unittest.plugins.codecoverage.CoverageResult;

config = jsondecode(fileread(config_file));
scope = config.sources(strcmp({config.sources.category}, 'statement_scope'));
source_files = fullfile(config.eeglab_root, {scope.path});
old_path = path;
old_directory = pwd;
old_visibility = get(0, 'DefaultFigureVisible');
cleanup = onCleanup(@() restore_environment(old_path, old_directory, old_visibility));
set(0, 'DefaultFigureVisible', 'off');
cd(config.test_root);
addpath(config.test_root, genpath(fullfile(config.test_root, 'unittesting_common')));
addpath(config.eeglab_root, genpath(fullfile(config.eeglab_root, 'functions')));
assert(strcmp(which('eeglab'), fullfile(config.eeglab_root, 'eeglab.m')), ...
    'eegprep:coverage:ShadowedSource', 'EEGLAB entry point does not resolve to the frozen source');
eeglab('nogui');
cd(config.test_root);
if ~isempty(config.support_paths)
    addpath(config.support_paths{:});
end
% The native variable, not an environment variable, controls option writes.
% Reuse the pinned icadefs unchanged, then isolate only its preference path.
options_directory = fullfile(config.output, 'options');
mkdir(options_directory);
if ~isempty(config.home_options)
    copyfile(config.home_options, fullfile(options_directory, 'eeg_options.m'));
end
stream = fopen(fullfile(options_directory, 'icadefs.m'), 'w');
fprintf(stream, 'run(''%s'');\nEEGOPTION_PATH = ''%s'';\n', ...
    strrep(which('icadefs'), '''', ''''''), strrep(options_directory, '''', ''''''));
fclose(stream);
addpath(options_directory, '-begin');

report = struct('metric', 'statement', 'matlab_version', version, 'execution_complete', false, ...
    'decision_coverage', 'Not collected by statement-level instrumentation', ...
    'validity_api', 'R2026a matlab.coverage.Result.Invalid (public-get, hidden property)', ...
    'worker_coverage', config.worker_coverage, 'current_test_files', {config.selected}, ...
    'selected_test_files', {config.selected}, ...
    'unselected_test_files', {setdiff(config.test_files, config.selected)}, ...
    'completed_test_files', {{}}, 'tests', [], 'files', [], ...
    'scope_file_count', numel(scope), 'measurable_file_count', 0, ...
    'denominator_complete', false, 'covered_statements', 0, 'total_statements', 0);
write_report(config.output, report);

suites = cell(1, numel(config.selected));
for index = 1:numel(config.selected)
    current = testsuite(fullfile(config.test_root, config.selected{index}));
    assert(~isempty(current), 'eegprep:coverage:EmptySuite', 'No native TestSuite cases found');
    suites{index} = current;
end
suite = [suites{:}];
report.discovered_test_cases = {suite.Name};
write_report(config.output, report);
format = CoverageResult;
runner = TestRunner.withTextOutput;
runner.addPlugin(XMLPlugin.producingJUnitFormat(fullfile(config.output, 'results.xml')));
runner.addPlugin(CodeCoveragePlugin.forFile(source_files, 'MetricLevel', 'statement', 'Producing', format));
results = runner.run(suite);
coverage = format.Result;
report = summarize(report, coverage, results, scope, source_files);
save(fullfile(config.output, 'coverage.mat'), 'coverage', 'results', '-v7.3');
report.execution_complete = true;
report.completed_test_files = config.selected;
report.current_test_files = {};
write_report(config.output, report);
assert(report.denominator_complete, 'eegprep:coverage:UnmeasurableSources', ...
    'Unmeasurable frozen source files; inspect report.json and native warnings');
assert(~any([results.Failed]) && ~any([results.Incomplete]), 'eegprep:coverage:NativeFailures', ...
    'Native failures/incomplete cases retained in report.json, coverage.mat and JUnit');
end

function report = summarize(report, coverage, results, scope, source_files)
[counts, details] = coverageSummary(coverage, 'statement');
function_counts = coverageSummary(coverage, 'function');
filenames = string({coverage.Filename});
files = repmat(struct('path', '', 'family', '', 'observed', false, ...
    'invalid', false, 'unmeasurable_reason', 'No native coverage result', ...
    'covered_statements', [], 'total_statements', [], ...
    'covered_functions', [], 'total_functions', [], 'details', []), numel(scope), 1);
for index = 1:numel(scope)
    files(index).path = scope(index).path;
    files(index).family = scope(index).family;
    match = find(strcmp(filenames, source_files{index}));
    if numel(match) == 1
        files(index).observed = true;
        files(index).invalid = coverage(match).Invalid;
        files(index).unmeasurable_reason = '';
        if files(index).invalid
            files(index).unmeasurable_reason = 'MATLAB marks source invalid; its 0/0 is not a measured denominator';
        end
        files(index).covered_statements = counts(match, 1);
        files(index).total_statements = counts(match, 2);
        files(index).covered_functions = function_counts(match, 1);
        files(index).total_functions = function_counts(match, 2);
        files(index).details = details(match);
    end
end
report.files = files;
measurable = [files.observed] & ~[files.invalid];
report.measurable_file_count = sum(measurable);
report.unmeasurable_files = {files(~measurable).path};
report.denominator_complete = all(measurable) && all(isfinite(counts(:)));
report.covered_statements = sum(counts(:, 1));
report.total_statements = sum(counts(:, 2));
report.tests = arrayfun(@(item) struct('name', item.Name, 'passed', item.Passed, ...
    'failed', item.Failed, 'incomplete', item.Incomplete, 'duration_seconds', item.Duration), results);
end

function write_report(directory, report)
file = fullfile(directory, 'report.json.tmp');
stream = fopen(file, 'w');
assert(stream ~= -1, 'Cannot open coverage report');
cleanup = onCleanup(@() fclose(stream));
fprintf(stream, '%s\n', jsonencode(report, PrettyPrint=true));
clear cleanup;
movefile(file, fullfile(directory, 'report.json'), 'f');
end

function restore_environment(old_path, old_directory, old_visibility)
path(old_path);
cd(old_directory);
set(0, 'DefaultFigureVisible', old_visibility);
end
