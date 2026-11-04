function labels = eegprep_test_pop_clust_kmeanscluster(preclustdata, component_numbers, cluster_count)
% Run command-line pop_clust kmeanscluster and return 1-based Cls labels.
component_numbers = component_numbers(:)';
count = size(preclustdata, 1);
study = struct();
study.filename = '';
study.filepath = '';
study.saved = 'no';
study.etc.preclust.preclustdata = preclustdata;
study.etc.preclust.clustlevel = 1;
study.etc.preclust.preclustparams = [];
study.cluster = struct('name', 'Parentcluster 1', 'sets', ones(1, count), 'comps', component_numbers);
study.cluster.parent = {};
study.cluster.child = {};
study = pop_clust(study, [], 'algorithm', 'kmeanscluster', 'clus_num', cluster_count);
labels = zeros(count, 1);
for index = 2:length(study.cluster)
    for comp = study.cluster(index).comps(:)'
        row = find(component_numbers == comp, 1, 'first');
        labels(row) = index - 1;
    end
end
end
