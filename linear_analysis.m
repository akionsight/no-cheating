% 1. Target the results directory
resultsDir = '/home/akionsight/Desktop/downloaded_data/results_7b';
matFiles = dir(fullfile(resultsDir, '*.mat'));

if isempty(matFiles)
    disp('No .mat files found in the specified directory.');
    return;
end

% 2. Initialize variables for cumulative tracking
cumulative_power = [];
total_files_processed = 0;
total_violations = 0;
fs = 1; % Sampling rate (1 step/unit)

% Set a fixed NFFT so all files output the same number of frequency bins
NFFT = 1024; 

fprintf('Starting batch processing...\n');

% 3. Loop through every .mat file
for k = 1:length(matFiles)

    currentFile = matFiles(k).name;
    filePath = fullfile(resultsDir, currentFile);

    try
        data = load(filePath);

        activations = data.acts; 
        labels = data.violated;

        [num_steps, num_features] = size(activations);
        power_spectra = zeros(NFFT, num_features);

        % 4. Compute FFT for each hidden dimension using fixed NFFT
        for i = 1:num_features
            Y = fft(activations(:, i), NFFT);
            % Normalize by NFFT instead of num_steps
            power_spectra(:, i) = abs(Y).^2 / NFFT; 
        end

        % 5. Isolate positive frequencies
        half_idx = floor(NFFT / 2) + 1;
        pos_freqs = (0:half_idx-1) * (fs / NFFT);
        pos_power = power_spectra(1:half_idx, :);

        % Calculate the mean power across all features for THIS file
        mean_file_power = mean(pos_power, 2);

        % 6. Accumulate the results safely
        if isempty(cumulative_power)
            cumulative_power = mean_file_power;
            common_freqs = pos_freqs; 
        else
            cumulative_power = cumulative_power + mean_file_power;
        end
        total_files_processed = total_files_processed + 1;

        % Tally total violations across all files
        total_violations = total_violations + sum(labels == 1);

    catch ME
        fprintf('  -> Error processing %s: %s\n', currentFile, ME.message);
    end
end

% 7. Calculate and Plot the Final Averages
if total_files_processed > 0
    average_power = cumulative_power / total_files_processed;

    figure('Name', 'Cumulative Spectral Analysis');
    plot(common_freqs, average_power, 'LineWidth', 2, 'Color', '#0072BD');
    title(sprintf('Average Power Spectral Density\n(Aggregated across %d rollouts)', total_files_processed));
    xlabel('Normalized Frequency (cycles/step)');
    ylabel('Average Power');
    grid on;

    fprintf('\n========================================\n');
    fprintf('CUMULATIVE BATCH RESULTS\n');
    fprintf('========================================\n');
    fprintf('Total files processed: %d\n', total_files_processed);
    fprintf('Total rule violations detected: %d\n', total_violations);
    fprintf('========================================\n');
else
    disp('No files were successfully processed.');
end