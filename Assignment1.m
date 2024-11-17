function main()
clusters.A.size = 250;
clusters.A.mean = [5; 10];
clusters.A.cov = [8, 0; 0, 4];
clusters.A.color = [0.2, 0.4, 0.7]; 
clusters.B.size = 250;
clusters.B.mean = [10; 15];
clusters.B.cov = [4, 0; 0, 8];
clusters.B.color = [0.8, 0.4, 0.4]; 
clusters.C.size = 150;
clusters.C.mean = [4; 11];
clusters.C.cov = [8, 4; 4, 35];
clusters.C.color = [0.4, 0.7, 0.4]; 
clusters.D.size = 250;
clusters.D.mean = [9; 16];
clusters.D.cov = [8, 0; 0, 8];
clusters.D.color = [0.7, 0.7, 0.2]; 
clusters.E.size = 200;
clusters.E.mean = [9; 6];
clusters.E.cov = [10, -6; -6, 21];
clusters.E.color = [0.6, 0.4, 0.7]; 
total_size = clusters.A.size + clusters.B.size + clusters.C.size + clusters.D.size + clusters.E.size;
fields = fieldnames(clusters);
for i = 1:length(fields)
label = fields{i};
clusters.(label).prior = clusters.(label).size / total_size;
end
for i = 1:length(fields)
label = fields{i};
params = clusters.(label);
samples = randn(params.size, 2);
[transformed_samples, eigenvalues, eigenvectors] = transformData(samples, params.mean, params.cov);
clusters.(label).samples = transformed_samples;
clusters.(label).eigenvalues = eigenvalues;
clusters.(label).eigenvectors = eigenvectors;
end
for i = 1:length(fields)
label = fields{i};
params = clusters.(label);
test_samples = randn(params.size, 2);
[transformed_test_samples, ~, ~] = transformData(test_samples, params.mean, params.cov);

clusters.(label).test_samples = transformed_test_samples;
end
fprintf('\nError Percentages for Clusters A and B:\n');
clusters_subset_AB = struct('A', clusters.A, 'B', clusters.B);
computeErrorsPerClass(clusters_subset_AB, 'MED');  
computeErrorsPerClass(clusters_subset_AB, 'GED');  
computeErrorsPerClass(clusters_subset_AB, 'MAP');  
computeErrorsPerClass(clusters_subset_AB, '5NN');  
computeErrorsPerClass(clusters_subset_AB, 'NN');  
fprintf('\nError Percentages for Clusters C, D, and E:\n');
clusters_subset_CDE = struct('C', clusters.C, 'D', clusters.D, 'E', clusters.E);
computeErrorsPerClass(clusters_subset_CDE, 'MED'); 
computeErrorsPerClass(clusters_subset_CDE, 'GED');  
computeErrorsPerClass(clusters_subset_CDE, 'MAP');  
computeErrorsPerClass(clusters_subset_CDE, '5NN');  
computeErrorsPerClass(clusters_subset_CDE, 'NN');   
figure('Position', [100, 100, 800, 600]);
labels = {'A', 'B'};
for i = 1:length(labels)
label = labels{i};
plot_cluster(clusters.(label).samples, clusters.(label).mean, clusters.(label).eigenvalues, ...
clusters.(label).eigenvectors, label, clusters.(label).color, false);
hold on;
end
title('Clusters A and B - MED, GED, and MAP Classification', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
[x_grid, y_grid] = meshgrid(linspace(-5, 20, 200), linspace(-10, 25, 200));
mu = {clusters.A.mean, clusters.B.mean};
clusters_subset = clusters_subset_AB;
plotMedGED(mu, clusters_subset, x_grid, y_grid);
plotMAP(clusters_subset, x_grid, y_grid);
[V_A, D_A] = eig(clusters.A.cov);
[V_B, D_B] = eig(clusters.B.cov);
plot_gaussian_contour(clusters.A.mean, clusters.A.cov, 'b');
plot_gaussian_contour(clusters.B.mean, clusters.A.cov, 'r');
draw_eigenvectors(clusters.A.mean, V_A, D_A, 'b');
draw_eigenvectors(clusters.B.mean, V_B, D_B, 'r');
axis equal;
legend({'A', 'B', 'MED', 'GED', 'MAP'}, 'FontSize', 12);
hold off;
figure('Position', [100, 100, 800, 600]);
for i = 1:length(labels)
label = labels{i};
plot_cluster(clusters.(label).samples, clusters.(label).mean, clusters.(label).eigenvalues, ...
clusters.(label).eigenvectors, label, clusters.(label).color, false);
hold on;
end
title('Clusters A and B - NN Classification', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
plotknn(1, clusters_subset, x_grid, y_grid);
axis equal;
legend({'A', 'B'}, 'FontSize', 12);
hold off;
figure('Position', [100, 100, 800, 600]);
labels = {'C', 'D', 'E'};
for i = 1:length(labels)
label = labels{i};
plot_cluster(clusters.(label).samples, clusters.(label).mean, clusters.(label).eigenvalues, ...
clusters.(label).eigenvectors, label, clusters.(label).color, false);
hold on;
end
title('Clusters C, D, and E - MED, GED, and MAP Classification', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
mu = {clusters.C.mean, clusters.D.mean, clusters.E.mean};
clusters_subset = clusters_subset_CDE;
plotMedGED(mu, clusters_subset, x_grid, y_grid);
plotMAP(clusters_subset, x_grid, y_grid);
[V_C, D_C] = eig(clusters.C.cov);
[V_D, D_D] = eig(clusters.D.cov);
[V_E, D_E] = eig(clusters.E.cov);
plot_gaussian_contour(clusters.C.mean, clusters.C.cov, 'g');
plot_gaussian_contour(clusters.D.mean, clusters.D.cov, 'y');
plot_gaussian_contour(clusters.E.mean, clusters.E.cov, 'm');
draw_eigenvectors(clusters.C.mean, V_C, D_C, 'g');
draw_eigenvectors(clusters.D.mean, V_D, D_D, 'y');
draw_eigenvectors(clusters.E.mean, V_E, D_E, 'm');
axis equal;
legend({'C', 'D', 'E', 'MED', 'GED', 'MAP'}, 'FontSize', 12);
hold off;
figure('Position', [100, 100, 800, 600]);
for i = 1:length(labels)
label = labels{i};
plot_cluster(clusters.(label).samples, clusters.(label).mean, clusters.(label).eigenvalues, ...
clusters.(label).eigenvectors, label, clusters.(label).color, false);
hold on;
end
title('Clusters C, D, and E - NN Classification', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
plotknn(1, clusters_subset, x_grid, y_grid);
axis equal;
legend({'C', 'D', 'E'}, 'FontSize', 12);
hold off;
end
function [transformed_samples, eigenvalues, eigenvectors] = transformData(samples, mean_vec, covariance)
[V, D] = eig(covariance);
[eigenvalues, idx] = sort(diag(D), 'descend');
eigenvectors = V(:, idx);
T = eigenvectors * diag(sqrt(eigenvalues));
transformed_samples = (T * samples')' + repmat(mean_vec', size(samples, 1), 1);
end
function plot_cluster(samples, mean_vec, eigenvalues, eigenvectors, label, color, withContour)
scatter(samples(:, 1), samples(:, 2), 30, 'MarkerEdgeColor', color, 'DisplayName', label, ...
'MarkerFaceColor', color, 'MarkerFaceAlpha', 0.5);
if ~withContour
return;
end
angle = atan2d(eigenvectors(2, 1), eigenvectors(1, 1));
ellipse_x_r = 2 * sqrt(eigenvalues(1));
ellipse_y_r = 2 * sqrt(eigenvalues(2));
t = linspace(0, 2 * pi, 100);
ellipse_x = ellipse_x_r * cos(t);
ellipse_y = ellipse_y_r * sin(t);
R = [cosd(angle), -sind(angle); sind(angle), cosd(angle)];
ellipse = R * [ellipse_x; ellipse_y];
plot(ellipse(1, :) + mean_vec(1), ellipse(2, :) + mean_vec(2), 'Color', color, 'LineWidth', 2);
hold on;
for i = 1:2
scale = sqrt(eigenvalues(i));
vector = eigenvectors(:, i) * scale;
quiver(mean_vec(1), mean_vec(2), vector(1), vector(2), 'Color', color, 'LineWidth', 1.5, 'MaxHeadSize', 0.5);
end
end
function plotMedGED(mu, clusters_subset, x_grid, y_grid)
[num_rows, num_cols] = size(x_grid);
grid_points = [x_grid(:), y_grid(:)];
distancesMED = zeros(length(grid_points), length(mu));
distancesGED = zeros(length(grid_points), length(mu));
cluster_labels = fieldnames(clusters_subset);
for k = 1:length(mu)
mu_k = mu{k};
distancesMED(:, k) = sqrt(sum((grid_points - mu_k').^2, 2));
distancesGED(:, k) = generalized_euclidean_distance(grid_points, mu_k, clusters_subset.(cluster_labels{k}).cov);
end
[~, closest_cluster_MED] = min(distancesMED, [], 2);
[~, closest_cluster_GED] = min(distancesGED, [], 2);
closest_cluster_MED = reshape(closest_cluster_MED, num_rows, num_cols);
closest_cluster_GED = reshape(closest_cluster_GED, num_rows, num_cols);
hold on;
contour(x_grid, y_grid, closest_cluster_MED, 'LineColor', [0, 0, 0], 'LineStyle', '--', 'LineWidth', 1);
contour(x_grid, y_grid, closest_cluster_GED, 'LineColor', [0.5, 0, 0.5], 'LineWidth', 1);
end
function plotMAP(clusters_subset, x_grid, y_grid)
[num_rows, num_cols] = size(x_grid);
grid_points = [x_grid(:), y_grid(:)];
labels = fieldnames(clusters_subset);
posterior_probs = zeros(length(grid_points), length(labels));
for k = 1:length(labels)
label = labels{k};
cluster = clusters_subset.(label);
mean_vec = cluster.mean;
cov_mat = cluster.cov;
prior = cluster.prior;
likelihood = mvnpdf(grid_points, mean_vec', cov_mat);
posterior_probs(:, k) = prior * likelihood;
end
[~, closest_cluster_MAP] = max(posterior_probs, [], 2);
closest_cluster_MAP = reshape(closest_cluster_MAP, num_rows, num_cols);
hold on;
contour(x_grid, y_grid, closest_cluster_MAP, 'LineColor', [0, 0.5, 0.5], 'LineStyle', '-.', 'LineWidth', 1);
end
function dist = generalized_euclidean_distance(x, mean, cov)
    diff = x - mean';
    inv_cov = inv(cov);
    dist = sqrt(sum((diff * inv_cov) .* diff, 2));
end
function plotknn(k, clusters_subset, x_grid, y_grid)
[num_rows, num_cols] = size(x_grid);
grid_points = [x_grid(:), y_grid(:)];
closest_clusters = zeros(length(grid_points), 1);
labels = fieldnames(clusters_subset);
all_samples = [];
all_labels = [];
for i = 1:length(labels)
label = labels{i};
cluster = clusters_subset.(label);
all_samples = [all_samples; cluster.samples];
all_labels = [all_labels; i * ones(size(cluster.samples, 1), 1)];
end
Mdl = KDTreeSearcher(all_samples);
idx = knnsearch(Mdl, grid_points, 'K', k);
for i = 1:length(grid_points)
k_nearest_labels = all_labels(idx(i, :));
counts = histcounts(k_nearest_labels, 0.5:1:length(labels) + 0.5);
[~, idx_max] = max(counts);
closest_clusters(i) = idx_max;
end
closest_clusters = reshape(closest_clusters, num_rows, num_cols);
hold on;
[~, h] = contourf(x_grid, y_grid, closest_clusters, 'LevelList', 1:length(labels), 'LineStyle', 'none');
colormap_colors = zeros(length(labels), 3);
for i = 1:length(labels)
colormap_colors(i, :) = clusters_subset.(labels{i}).color;
end
colormap(gca, colormap_colors);
set(h, 'FaceAlpha', 0.3);
contour(x_grid, y_grid, closest_clusters, 'LineColor', [0, 0.5, 0.5], 'LineWidth', 1);
children = get(gca, 'Children');
set(gca, 'Children', flipud(children));
end
function computeErrorsPerClass(clusters_subset, classifier)
labels = fieldnames(clusters_subset);
label_to_index = containers.Map(labels, 1:length(labels));
total_errors = 0;
total_samples = 0;
fprintf('\n');

switch classifier
    case 'MED'
        fprintf('Error percentages for MED classifier:\n');
        for i = 1:length(labels)
            label = labels{i};
            cluster = clusters_subset.(label);
            num_samples = size(cluster.test_samples, 1);
            distances = zeros(num_samples, length(labels));
            for j = 1:length(labels)
                other_label = labels{j};
                other_cluster = clusters_subset.(other_label);
                distances(:, j) = sqrt(sum((cluster.test_samples - other_cluster.mean').^2, 2));
            end
            [~, predicted_indices] = min(distances, [], 2);
            true_index = label_to_index(label);
            num_errors = sum(predicted_indices ~= true_index);
            error_percentage = (num_errors / num_samples) * 100;
            fprintf('Class %s error (%s): %.2f%%\n', label, classifier, error_percentage);
            displayPredictionBreakdown(predicted_indices, labels, label);
            total_errors = total_errors + num_errors;
            total_samples = total_samples + num_samples;
        end
    case 'GED'
        fprintf('Error percentages for GED classifier:\n');
        for i = 1:length(labels)
            label = labels{i};
            cluster = clusters_subset.(label);
            num_samples = size(cluster.test_samples, 1);
            distances = zeros(num_samples, length(labels));
            for j = 1:length(labels)
                other_label = labels{j};
                other_cluster = clusters_subset.(other_label);
                distances(:, j) = generalized_euclidean_distance(cluster.test_samples, other_cluster.mean, other_cluster.cov);
            end
            [~, predicted_indices] = min(distances, [], 2);
            true_index = label_to_index(label);
            num_errors = sum(predicted_indices ~= true_index);
            error_percentage = (num_errors / num_samples) * 100;
            fprintf('Class %s error (%s): %.2f%%\n', label, classifier, error_percentage);
            displayPredictionBreakdown(predicted_indices, labels, label);
            total_errors = total_errors + num_errors;
            total_samples = total_samples + num_samples;
        end
    case 'MAP'
        fprintf('Error percentages for MAP classifier:\n');
        for i = 1:length(labels)
            label = labels{i};
            cluster = clusters_subset.(label);
            num_samples = size(cluster.test_samples, 1);
            posterior_probs = zeros(num_samples, length(labels));
            for j = 1:length(labels)
                other_label = labels{j};
                other_cluster = clusters_subset.(other_label);
                prior = other_cluster.prior;
                likelihood = mvnpdf(cluster.test_samples, other_cluster.mean', other_cluster.cov);
                posterior_probs(:, j) = prior * likelihood;
            end
            [~, predicted_indices] = max(posterior_probs, [], 2);
            true_index = label_to_index(label);
            num_errors = sum(predicted_indices ~= true_index);
            error_percentage = (num_errors / num_samples) * 100;
            fprintf('Class %s error (%s): %.2f%%\n', label, classifier, error_percentage);
            displayPredictionBreakdown(predicted_indices, labels, label);
            total_errors = total_errors + num_errors;
            total_samples = total_samples + num_samples;
        end
    case '5NN'
        fprintf('Error percentages for 5-NN classifier:\n');
        all_train_samples = [];
        all_train_labels = [];
        for j = 1:length(labels)
            other_label = labels{j};
            other_cluster = clusters_subset.(other_label);
            all_train_samples = [all_train_samples; other_cluster.samples];
            all_train_labels = [all_train_labels; j * ones(size(other_cluster.samples, 1), 1)];
        end
        Mdl = KDTreeSearcher(all_train_samples);
        for i = 1:length(labels)
            label = labels{i};
            cluster = clusters_subset.(label);
            num_samples = size(cluster.test_samples, 1);
            idx = knnsearch(Mdl, cluster.test_samples, 'K', 5);
            predicted_labels = mode(all_train_labels(idx), 2);
            true_label = label_to_index(label);
            num_errors = sum(predicted_labels ~= true_label);
            error_percentage = (num_errors / num_samples) * 100;
            fprintf('Class %s error (5NN): %.2f%%\n', label, error_percentage);
            displayPredictionBreakdown(predicted_labels, labels, label);
             total_errors = total_errors + num_errors;
            total_samples = total_samples + num_samples;
        end
    case 'NN'
        fprintf('Error percentages for 1-NN classifier:\n');
        all_train_samples = [];
        all_train_labels = [];
        for j = 1:length(labels)
            other_label = labels{j};
            other_cluster = clusters_subset.(other_label);
            all_train_samples = [all_train_samples; other_cluster.samples];
            all_train_labels = [all_train_labels; j * ones(size(other_cluster.samples, 1), 1)];
        end
        Mdl = KDTreeSearcher(all_train_samples);
        for i = 1:length(labels)
            label = labels{i};
            cluster = clusters_subset.(label);
            num_samples = size(cluster.test_samples, 1);
            idx = knnsearch(Mdl, cluster.test_samples, 'K', 1);
            predicted_labels = all_train_labels(idx);
            true_label = label_to_index(label);
            num_errors = sum(predicted_labels ~= true_label);
            error_percentage = (num_errors / num_samples) * 100;
            fprintf('Class %s error (%s): %.2f%%\n', label, classifier, error_percentage);
   
            displayPredictionBreakdown(predicted_labels, labels, label);
            
            total_errors = total_errors + num_errors;
            total_samples = total_samples + num_samples;
        end
    otherwise
        error('Unknown classifier: %s', classifier);
end

overall_error = (total_errors / total_samples) * 100;
fprintf('Overall error (%s): %.2f%%\n', classifier, overall_error);
end
function displayPredictionBreakdown(predicted_indices, labels, true_label)
fprintf('  Class %s predictions:\n', true_label);
for j = 1:length(labels)
    num_predicted = sum(predicted_indices == j);
    fprintf('    Predicted %s: %d points\n', labels{j}, num_predicted);
end
end
function plot_gaussian_contour(mu, cov, color)
    t = linspace(0, 2*pi, 100);
    circle = [cos(t); sin(t)];
    [V, D] = eig(cov);
    ellipse = V * sqrt(D) * circle;
    plot(ellipse(1,:) + mu(1), ellipse(2,:) + mu(2), color, 'LineWidth', 1.5);
end
function draw_eigenvectors(mu, V, D, color)
    for i = 1:2
        eigenvector = V(:,i) * sqrt(D(i,i)) * 2;
        quiver(mu(1), mu(2), eigenvector(1), eigenvector(2), color, ...
               'LineWidth', 0.5, 'MaxHeadSize', 0.2, 'AutoScale', 'off', 'ShowArrowHead', 'on');
    end
end
main();



