function VOCopts = get_voc_opts(path, dataset, test_set)

tmp = pwd;
cd(path);
try
  addpath('VOCcode');
  VOCopts = VOCinit(dataset, test_set);
catch
  rmpath('VOCcode');
  cd(tmp);
  error(sprintf('VOCcode directory not found under %s', path));
end
rmpath('VOCcode');
cd(tmp);
