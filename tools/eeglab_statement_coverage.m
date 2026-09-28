function eeglab_statement_coverage(config_file)
% Native TestRunner statement coverage, with a fixed, independently frozen scope.
import matlab.unittest.TestRunner;
import matlab.unittest.plugins.CodeCoveragePlugin;
import matlab.unittest.plugins.XMLPlugin;
import matlab.unittest.plugins.codecoverage.CoverageResult;

config = jsondecode(fileread(config_file));
if ~isempty(fieldnames(config.unmeasurable_approval))
    assert(strcmp(config.unmeasurable_approval.matlab_version, version), ...
        'eegprep:coverage:DifferentRelease', 'Unmeasurable-source approval requires its documented MATLAB release');
end
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

original_selected = cellstr(string(config.selected));
test_files = fullfile(config.test_root, original_selected);
additional_labels = {};
for index = 1:numel(config.additional_test_sources)
    addition = config.additional_test_sources(index);
    test_files{end+1} = fullfile(config.output, addition.snapshot);
    additional_labels{end+1} = addition.snapshot;
end
selected_labels = [reshape(original_selected, 1, []), additional_labels];

report = struct('metric', 'statement', 'matlab_version', version, 'execution_complete', false, ...
    'decision_coverage', 'Not collected by statement-level instrumentation', ...
    'validity_api', 'R2026a matlab.coverage.Result.Invalid (public-get, hidden property)', ...
    'worker_coverage', config.worker_coverage, 'current_test_files', {selected_labels}, ...
    'selected_test_files', {selected_labels}, ...
    'additional_test_sources', config.additional_test_sources, ...
    'baseline', config.baseline, ...
    'unmeasurable_approval', config.unmeasurable_approval, ...
    'unselected_test_files', {setdiff(config.test_files, original_selected)}, ...
    'completed_test_files', {{}}, 'tests', [], 'files', [], ...
    'scope_file_count', numel(scope), 'measurable_file_count', 0, ...
    'denominator_complete', false, 'covered_statements', 0, 'total_statements', 0);
write_report(config.output, report);

suites = cell(1, numel(test_files));
for index = 1:numel(test_files)
    current = testsuite(test_files{index});
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
batch_counts = coverageSummary(coverage, 'statement');
report.batch_covered_statements = sum(batch_counts(:, 1));
report.newly_covered_statements = report.batch_covered_statements;
baseline_failed = false;
if ~isempty(fieldnames(config.baseline))
    assert(strcmp(config.baseline.matlab_version, version), ...
        'eegprep:coverage:DifferentRelease', 'Coverage union requires the same MATLAB release');
    previous = load(fullfile(config.baseline.directory, 'coverage.mat'), 'coverage');
    assert(isequal(sort(string({previous.coverage.Filename})), sort(string({coverage.Filename}))), ...
        'eegprep:coverage:DifferentScope', 'Coverage union requires the same source filenames');
    % Native coverage union deduplicates statements; scalar count addition does not.
    coverage = previous.coverage + coverage;
    union_counts = coverageSummary(coverage, 'statement');
    report.newly_covered_statements = sum(union_counts(:, 1)) - config.baseline.covered_statements;
    baseline_failed = config.baseline.native_failures;
end
report = summarize(report, coverage, results, scope, source_files);
save(fullfile(config.output, 'coverage.mat'), 'coverage', 'results', '-v7.3');
report.execution_complete = true;
report.completed_test_files = selected_labels;
report.current_test_files = {};
write_report(config.output, report);
assert(report.approved_denominator_complete, 'eegprep:coverage:UnmeasurableSources', ...
    'Unmeasurable frozen source files; inspect report.json and native warnings');
assert(~any([results.Failed]) && ~any([results.Incomplete]) && ~baseline_failed, ...
    'eegprep:coverage:NativeFailures', ...
    'Native failures/incomplete cases retained in current or baseline report.json, coverage.mat and JUnit');
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
approved = {};
if ~isempty(fieldnames(report.unmeasurable_approval))
    approved = {report.unmeasurable_approval.sources.path};
end
report.approved_unmeasurable_files = intersect(report.unmeasurable_files, approved);
report.unapproved_unmeasurable_files = setdiff(report.unmeasurable_files, approved);
report.approved_denominator_complete = all([files.observed]) && ...
    isempty(report.unapproved_unmeasurable_files) && all(isfinite(counts(:)));
report.covered_statements = sum(counts(:, 1));
report.total_statements = sum(counts(:, 2));
report.tests = arrayfun(@(item) struct('name', item.Name, 'passed', item.Passed, ...
    'failed', item.Failed, 'incomplete', item.Incomplete, 'duration_seconds', item.Duration), ...
    results, 'UniformOutput', false);
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
