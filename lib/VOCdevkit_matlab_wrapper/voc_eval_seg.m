function voc_eval_seg(path, comp_id, year, test_set, output_dir)

dataset = strcat('VOC', year);
VOCopts = get_voc_opts(path, dataset, test_set);
addpath(fullfile(VOCopts.datadir, 'VOCcode'));

[ acc, avacc, conf, ~ ] = VOCevalseg(VOCopts, comp_id);

if ~exist(output_dir)
	mkdir(output_dir)
end
save([output_dir '/seg_res.mat'], ...
     'acc', 'avacc', 'conf');

fprintf('\n~~~~~~~~~~~~~~~~~~~~\n');
fprintf('Results are saved\n');
fprintf('~~~~~~~~~~~~~~~~~~~~\n');
acc
fprintf('~~~~~~~~~~~~~~~~~~~~\n');
avacc
fprintf('~~~~~~~~~~~~~~~~~~~~\n');
