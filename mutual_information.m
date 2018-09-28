function [mutualInformation, jointEntropy, entropy1, entropy2] = mutual_information(X,Y,nbins)
    % inputs: X, Y, matrices of the same size
    %         nbins: number of bins for double data sets, default:
    %         log2(length(X(:)))
    % output:
    %   mutual information of X and Y
    %   joint entropy of X and Y
    %   entropy of X
    %   entropy of Y
    %
    % source: https://stackoverflow.com/a/23691992/1209885
    %
    % Edited by: Keivan Hassani Monfared, k1monfared@gmail.com
    
    if nargin < 3
        nbins = log2(length(X(:))); % total number of elements
        if nargin < 2
            Y = X;
        end
    end    
    
    edgex = min(X):(max(X)-min(X))/nbins:max(X)+(max(X)-min(X))/nbins; % creat bins
    edgey = min(Y):(max(Y)-min(Y))/nbins:max(Y)+(max(Y)-min(Y))/nbins; % creat bins
    % and extra bin is added at the end to ensure the numbers in the top
    % bin are accounted for
    im1 = discretize(X,edgex); % bin X
    im2 = discretize(Y,edgey); % bin Y
    
    % accumarray assumes that you are trying to index into the output array 
    % using integers, but we can still certainly accomplish what we want 
    % with this small bump in the road. What you would do is simply assign 
    % each floating point value in both images to have a unique ID. You 
    % would thus use accumarray with these IDs instead. To facilitate this 
    % ID assigning, use unique - specifically the third output from the 
    % function. You would take each of the images, put them into unique and 
    % make these the indices to be input into accumarray. In other words, 
    % do this instead:
    
    [~,~,indrow] = unique(double(im1(:)));
    [~,~,indcol] = unique(double(im2(:)));
    
    % Note that with indrow and indcol, we are directly assigning the third 
    % output of unique to these variables and then using the same joint 
    % entropy code that we computed earlier. We also don't have to offset 
    % the variables by 1 as we did previously because unique will assign 
    % IDs starting at 1.
    
    % You can actually calculate the histograms or probability 
    % distributions for each image individually using the joint probability 
    % matrix. If you wanted to calculate the histograms / probability 
    % distributions for the first image, you would simply accumulate all of 
    % the columns for each row. To do it for the second image, you would 
    % simply accumulate all of the rows for each column. As such, you can do:
    jointHistogram = accumarray([indrow indcol], 1);
    jointProb = jointHistogram / numel(indrow);
    indNoZero = jointHistogram ~= 0;
    jointProb1DNoZero = jointProb(indNoZero);
    jointEntropy = -sum(jointProb1DNoZero.*log2(jointProb1DNoZero));

    histogramImage1 = sum(jointHistogram, 1);
    histogramImage2 = sum(jointHistogram, 2);
    
    %After, you can calculate the entropy of both of these by yourself. To 
    % double check, make sure you turn both of these into PDFs, then 
    % compute the entropy using the standard equation (like above).

    % To finally compute Mutual Information, you're going to need the 
    % entropy of the two images. You can use MATLAB's built-in entropy 
    % function, but this assumes that there are 256 unique levels. You 
    % probably want to apply this for the case of there being N distinct 
    % levels instead of 256, and so you can use what we did above with the 
    % joint histogram, then computing the histograms for each image in the 
    % aside code above, and then computing the entropy for each image. You 
    % would simply repeat the entropy calculation that was used jointly, 
    % but apply it to each image individually:
    
    %// Find non-zero elements for first image's histogram
    indNoZero = histogramImage1 ~= 0;

    %// Extract them out and get the probabilities
    prob1NoZero = histogramImage1(indNoZero);
    prob1NoZero = prob1NoZero / sum(prob1NoZero);

    %// Compute the entropy
    entropy1 = -sum(prob1NoZero.*log2(prob1NoZero));

    %// Repeat for the second image
    indNoZero = histogramImage2 ~= 0;
    prob2NoZero = histogramImage2(indNoZero);
    prob2NoZero = prob2NoZero / sum(prob2NoZero);
    entropy2 = -sum(prob2NoZero.*log2(prob2NoZero));

    %// Now compute mutual information
    mutualInformation = entropy1 + entropy2 - jointEntropy;
end