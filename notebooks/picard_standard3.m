% Runs the Picard algorithm
%
% The algorithm is detailed in::
%
%     Pierre Ablin, Jean-Francois Cardoso, and Alexandre Gramfort
%     Faster independent component analysis by preconditioning with Hessian
%     approximations
%     IEEE Transactions on Signal Processing, 2018
%     https://arxiv.org/abs/1706.08171
%
% Parameters
% ----------
% X : array, shape (N, T)
%     Matrix containing the signals that have to be unmixed. N is the
%     number of signals, T is the number of samples. X has to be centered
%
% m : int
%     Size of L-BFGS's memory. Typical values for m are in the range 3-15
%
% maxiter : int
%     Maximal number of iterations for the algorithm
%
% precon : 1 or 2
%     Chooses which Hessian approximation is used as preconditioner.
%     1 -> H1
%     2 -> H2
%     H2 is more costly to compute but can greatly accelerate convergence
%     (See the paper for details).
%
% tol : float
%     tolerance for the stopping criterion. Iterations stop when the norm
%     of the gradient gets smaller than tol.
%
% lambda_min : float
%     Constant used to regularize the Hessian approximations. The
%     eigenvalues of the approximation that are below lambda_min are
%     shifted to lambda_min.
%
% ls_tries : int
%     Number of tries allowed for the backtracking line-search. When that
%     number is exceeded, the direction is thrown away and the gradient
%     is used instead.
%
% verbose : boolean
%     If true, prints the informations about the algorithm.
%
% Returns
% -------
% Y : array, shape (N, T)
%     The estimated source matrix
%
% W : array, shape (N, N)
%     The estimated unmixing matrix, such that Y = WX.
%
% Modified by Arnaud Delorme to match the Python implementation
% Changes compared to picard_standard to match Python
% - Matlab used the standard logistic distribution while python uses the
% log-cosh (hyperbolic-secant) mode (preferable according to AI)
% - The hessian regularization returned different numbers on the diagonal
% (the old hessian is commented in this code)

% Authors: Pierre Ablin <pierre.ablin@inria.fr>
%          Alexandre Gramfort <alexandre.gramfort@inria.fr>
%          Jean-Francois Cardoso <cardoso@iap.fr>
%          Arnaud Delorme <arno@ucsd.edu>
%
% License: BSD (3-clause)

function [Y, W] = picard_standard3(X, m, maxiter, precon, tol, lambda_min, ls_tries, verbose)

    [N, T] = size(X);
    W = eye(N);
    Y = X;
    s_list = {};
    y_list = {};
    r_list = {};
    current_loss = loss(Y, W);
    fprintf('MATLAB initial simple_loss = %.10g\n', current_loss);

    for n_top = 1:maxiter
        % Compute the score function
        psiY = tanh(Y);
        G    = psiY * Y' / T;
        % Compute the relative gradient
        G    = G - eye(N);

        % Stopping criterion
        G_norm = max(abs(G), [], 'all');
        if G_norm < tol
            break
        end

        % Update the memory
        if n_top > 1
            s_list{end+1} = direction;
            y = G - G_old;
            y_list{end+1} = y;
            r_list{end+1} = 1 / sum(direction .* y, 'all');
            if numel(s_list) > m
                s_list = s_list(2:end);
                y_list = y_list(2:end);
                r_list = r_list(2:end);
            end
        end
        G_old = G;

        % Find the L-BFGS direction
        direction = l_bfgs_direction(Y, psiY, G, s_list, y_list, r_list, precon, lambda_min);
        % Do a line_search in that direction:
        [converged, new_Y, new_W, new_loss, direction] = line_search(Y, W, direction, current_loss, ls_tries, verbose);
        if ~converged
            direction = -G;
            s_list = {};
            y_list = {};
            r_list = {};
            [~, new_Y, new_W, new_loss, direction] = line_search(Y, W, direction, current_loss, 10, false);
        end

        Y = new_Y;
        W = new_W;
        current_loss = new_loss;

        if verbose
            fprintf('iteration %d, gradient norm = %.6g loss = %.6g\n', n_top, G_norm, current_loss);
        end
    end

function loss_val = loss(Y, W)
    %
    % Computes the loss function for Y, W
    %
    N = size(Y, 1);
    loss_val = -log(det(W));
    for k = 1:N
        y = Y(k, :);
        loss_val = loss_val + mean( abs(y) + log1p( exp(-2*abs(y)) ) );
        % loss_val = loss + mean(abs(y) + 2. * log1p(exp(-abs(y)))); % old loss
    end

function direction = l_bfgs_direction(Y, psiY, G, s_list, y_list, r_list, precon, lambda_min)
    q = G;
    a_list = {};
    for ii = 1:numel(s_list)
        s = s_list{end-ii+1};
        y = y_list{end-ii+1};
        r = r_list{end-ii+1};
        alpha = r * sum(s .* q, 'all');
        a_list{end+1} = alpha;
        q = q - alpha * y;
    end
    z = solve_hessian(q, Y, psiY, precon, lambda_min);
    for ii = 1:numel(s_list)
        s = s_list{ii};
        y = y_list{ii};
        r = r_list{ii};
        alpha = a_list{end-ii+1};
        beta  = r * sum(y .* z, 'all');
        z = z + (alpha - beta) * s;
    end
    direction = -z;

function a = regularize_hessian(a, lam)
    %REGULARIZE_HESSIAN Enforce minimum eigenvalue lam on Hessian approx a
    %   a     – N×N Hessian approximation matrix
    %   lam   – scalar minimum allowed eigenvalue
    %   returns a with any off-diagonal entry adjusted so that all
    %   eigenvalues of the symmetric part are at least lam

    N = size(a, 1);
    % compute the smaller eigenvalue of each 2×2 block
    e = 0.5*(a + a' - sqrt((a - a').^2 + 4));
    % find entries where that eigenvalue is below lam (excluding diagonal)
    mask = e < lam;
    mask(1:(N+1):N*N) = false;
    [i, j] = find(mask);
    idx = sub2ind([N, N], i, j);
    % shift those entries so that their block eigenvalue equals lam
    a(idx) = a(idx) + (lam - e(idx));

function out = solve_hessian(G, Y, psiY, precon, lambda_min)
    [N, T] = size(Y);
    % psidY = (- thY.^2 + 1.) / 2.; % old code
    psidY   = 1 - psiY.^2;
    % Build the diagonal of the Hessian, a.
    Y2      = Y.^2;
    if precon == 2
        a = (psidY * Y2') / T;
    elseif precon == 1
        sigma2     = mean(Y2, 2);
        psid_mean  = mean(psidY, 2);
        a          = psid_mean * sigma2';
        diag_term  = mean(mean(Y2 .* psidY)) + 1;
        a(1:(N+1):N*N) = diag_term;
    else
        error('precon must be 1 or 2');
    end

    if 1
        a = regularize_hessian(a, lambda_min);
    else
        % old code, values differ on the diagonal
        % Compute the eigenvalues of the Hessian
        eigs = 0.5 * (a + a' - sqrt((a - a').^2 + 4));
        pb   = eigs < lambda_min;
        pb(1:(N+1):N*N) = false;
        [i_pb, j_pb]   = find(pb);
        a(i_pb, j_pb)  = a(i_pb, j_pb) + lambda_min - eigs(i_pb, j_pb);
    end

    % Invert the transform
    out  = (G .* a' - G') ./ (a .* a' - 1);


function [converged, Y_new, W_new, new_loss, rel_step] = line_search(Y, W, direction, current_loss, ls_tries, verbose)
    %
    % Performs a backtracking line search, starting from Y and W, in the
    % direction direction. I
    %
    N = size(Y, 1);
    projected_W = direction * W;
    alpha = 1.;
    for tmp=1:ls_tries
        Y_new = (eye(N) + alpha * direction) * Y;
        W_new = W + alpha * projected_W;
        new_loss = loss(Y_new, W_new);
        if new_loss < current_loss
            converged = true;
            rel_step = alpha * direction;
            return
        end
        alpha = alpha / 2.;
    end
    if verbose
        fprintf('line search failed, falling back to gradient.\n');
    end
    converged = false;
    rel_step = alpha * direction;
