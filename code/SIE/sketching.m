mpath = which('sketching');
mpath(end-11:end) = []
cd(mpath)
pwd

fileFolder=fullfile('SmoothImage');
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
     Im = imread(['SmoothImage/' filename]);
     I = PencilDrawing(Im, 7, 0.1, 10, 1.0, 0);
     fprintf([filename ' in SmoothImage Saved\n'])
     imwrite(I, ['SketchImage/' filename])
end
