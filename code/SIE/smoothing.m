mpath = which('smoothing');
mpath(end-11:end) = []
cd(mpath)
pwd

fileFolder=fullfile('ResizedImage');
dirOutput=dir(fullfile(fileFolder,'*'));
fileNames={dirOutput.name}';
length = size(fileNames);
filesize = length(1, 1);


for i = 1 : 1 : filesize
     if strcmp(fileNames{i, 1},'.') || strcmp(fileNames{i, 1},'..') || strcmp(fileNames{i, 1},'.DS_Store')
         continue
     end
     filename = fileNames{i, 1};
     fprintf(['\n>>>>>>>>>>>>>\nProcessing ' filename '\n'])
     Im = imread(['ResizedImage/' filename]);
     S = L0Smoothing(Im, 0.02, 1.2);
     fprintf([filename ' Saved\n'])
     imwrite(S, ['SmoothImage/' filename])
end
