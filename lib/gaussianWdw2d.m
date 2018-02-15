function [window] = gaussianWdw2d(mu_x, sigma_x, mu_y, sigma_y, x, y)

    %window = exp(-(x - mu_x).^2 / 2 / sigma_x / sigma_x) .* exp(-(y - mu_y).^2 / 2 / sigma_y / sigma_y) ;
    window = mtimes(exp(-(x - mu_x).^2 / 2 / sigma_x / sigma_x)', exp(-(y - mu_y).^2 / 2 / sigma_y / sigma_y)) ;
end