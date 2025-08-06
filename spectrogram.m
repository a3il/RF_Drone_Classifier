numChannels = 1;
samplingRate = 200e6;
duration = 15;
frequencyResolution = 0.1;
chunkSize = 5e4;

folderPath = 'E:\Work\Preprocessing DataSet\Final Rev\All csv';
fileList = dir(fullfile(folderPath, '*.csv'));
numFiles = numel(fileList);

spectrogramFolder = 'C:\Users\adila\OneDrive\Desktop\specc';
if ~exist(spectrogramFolder, 'dir')
    mkdir(spectrogramFolder);
end

windowLength = round(samplingRate / frequencyResolution);

for i = 1:numFiles
    filename = fullfile(folderPath, fileList(i).name);
    data = csvread(filename);
    
    [~, name, ~] = fileparts(filename);
    
    data = reshape(data, [], numChannels);
    
    data = single(data);
    
    numSamples = round(duration * samplingRate);
    
    numChunks = ceil(numSamples / chunkSize);
    
    spectrogramData = [];
    
    for j = 1:numChunks
        startIndex = (j - 1) * chunkSize + 1;
        endIndex = min(startIndex + chunkSize - 1, numSamples);
        
        if endIndex > size(data, 1)
            endIndex = size(data, 1);
        end
        
        chunkData = data(startIndex:endIndex, :);
        
        chunkWindowLength = min(windowLength, size(chunkData, 1));
        
        overlapLength = round(chunkWindowLength * 0.9);
        if overlapLength >= chunkWindowLength
            overlapLength = chunkWindowLength - 1;
        end
        
        chunkSpectrogram = zeros(windowLength, size(chunkData, 2));
        
        for k = 1:numChannels
            channelData = chunkData(:, k);
            
            [~, ~, S] = spectrogram(channelData, hamming(chunkWindowLength), overlapLength, chunkWindowLength, samplingRate);
            
            chunkSpectrogram(:, k) = abs(S(:, 1));
        end
        
        spectrogramData = [spectrogramData; chunkSpectrogram];
    end
    
    figure('Visible', 'off');
    imagesc(log10(spectrogramData));
    colormap(jet);
    axis off;
    
    spectrogramFilename = fullfile(spectrogramFolder, [name, '.png']);
    saveas(gcf, spectrogramFilename);
    
    close(gcf);
end
