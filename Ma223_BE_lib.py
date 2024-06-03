from numba import jit, prange
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

@jit(nopython=True)
def approx_f(F_r, F_r1, i, j):

    m, n = F_r.shape

    if i == 0:
        i_min, i_max = 0, 1
    elif i == m - 1:
        i_min, i_max = m - 2, m - 1
    else:
        i_min, i_max = i - 1, i + 1

    if j == 0:
        j_min, j_max = 0, 1
    elif j == n - 1:
        j_min, j_max = n - 2, n - 1
    else:
        j_min, j_max = j - 1, j + 1

    df_dx = (F_r1[i_max, j] - F_r1[i_min, j] + F_r[i_max, j] - F_r[i_min, j]) / 4
    df_dy = (F_r1[i, j_max] - F_r1[i, j_min] + F_r[i, j_max] - F_r[i, j_min]) / 4
    df_dt = F_r1[i, j] - F_r[i, j]

    return df_dx, df_dy, df_dt

def plot_derivate(F_r, F_r1):

    F_r = cv2.imread(F_r, cv2.IMREAD_GRAYSCALE).astype(float)
    F_r1 = cv2.imread(F_r1, cv2.IMREAD_GRAYSCALE).astype(float)

    m, n = F_r.shape
    dx = np.zeros((m, n))
    dy = np.zeros((m, n))
    dt = np.zeros((m, n))

    for i in range(m):
        for j in range(n):
            dx[i, j], dy[i, j], dt[i, j] = approx_f(F_r, F_r1, i, j)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(dx, cmap='binary')
    plt.title('∂f/∂x')

    plt.subplot(1, 3, 2)
    plt.imshow(dy, cmap='binary')
    plt.title('∂f/∂y')

    plt.subplot(1, 3, 3)
    plt.imshow(dt, cmap='binary')
    plt.title('∂f/∂t')

    plt.show()

def MC_flux_calc(F_r, F_r1, h=3):

    m, n = F_r.shape
    V1 = np.zeros((m, n))
    V2 = np.zeros((m, n))
    Lambda = np.zeros((m, n))

    for i in range(h, m - h):
        for j in range(h, n - h):
            A = []
            b = []

            for p in range(-h, h + 1):
                for q in range(-h, h + 1):
                    df_dx, df_dy, df_dt = approx_f(F_r, F_r1, i + p, j + q)
                    A.append([df_dx, df_dy])
                    b.append(-df_dt)

            A = np.array(A)
            b = np.array(b).reshape(-1, 1)

            if np.linalg.matrix_rank(A) == 2:
                ATA = A.T @ A
                ATb = A.T @ b
                Lambda_min = np.min(np.linalg.eigvals(ATA))
                v = np.linalg.solve(ATA, ATb)
                V1[i, j] = v[0]
                V2[i, j] = v[1]
                Lambda[i, j] = Lambda_min
            else:
                V1[i, j] = 0
                V2[i, j] = 0
                Lambda[i, j] = 0

    return V1, V2, Lambda

def MC_flux_calc_v1(F_r, F_r1, h=3, lambda_param=10):

    m, n = F_r.shape
    V1 = np.zeros((m, n))
    V2 = np.zeros((m, n))
    Lambda = np.zeros((m, n))

    for i in range(h, m - h):
        for j in range(h, n - h):
            A = []
            b = []
            for p in range(-h, h + 1):
                for q in range(-h, h + 1):
                    dxF, dyF, dtF = approx_f(F_r, F_r1, i + p, j + q)
                    A.append([dxF, dyF])
                    b.append(-dtF)
            A = np.array(A)
            b = np.array(b)
            ATA = A.T @ A
            ATb = A.T @ b
            if np.linalg.matrix_rank(ATA) == 2:
                v = np.linalg.solve(ATA + lambda_param * np.eye(2), ATb)
                V1[i, j] = v[0]
                V2[i, j] = v[1]
                Lambda[i, j] = np.min(np.linalg.eigvals(ATA))

    return V1, V2, Lambda

@jit(nopython=True)
def comb(n, k):

    if k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    c = 1
    for i in range(k):
        c = c * (n - i) // (i + 1)
    return c


def matrice_poids(h, p):

    size = 2 * h + 1
    W = np.zeros((size ** 2, size ** 2))
    weights = np.zeros(size)

    for k in range(size):
        coef = comb(2 * h, k) * (p ** k) * ((1 - p) ** (2 * h - k))
        weights[k] = coef

    weights /= np.sum(weights)
    plt.figure(figsize=(12, 9))
    plt.title(f"Densité discrète de la loi binomiale pour p={p}")
    plt.stem(weights)
    plt.xlabel('k')
    plt.ylabel('Probabilité')
    plt.show()
    for i in range(size ** 2):
        W[i, i] = weights[i % size]

    return W


def MC_flux_calc_v2(F_r, F_r1, h=3, p=1.0):

    F_r = F_r.astype(np.float64)
    F_r1 = F_r1.astype(np.float64)

    m, n = F_r.shape
    V1 = np.zeros((m, n))
    V2 = np.zeros((m, n))
    Lambda = np.zeros((m, n))

    W = matrice_poids(h, p)

    for i in prange(h, m - h):
        for j in prange(h, n - h):
            A = []
            b = []
            for k in prange(-h, h + 1):
                for l in prange(-h, h + 1):
                    dx, dy, dt = approx_f(F_r, F_r1, i + k, j + l)
                    A.append([dx, dy])
                    b.append(-dt)

            A = np.array(A)
            b = np.array(b)
            A_w = W @ A
            b_w = W @ b

            ATA = A.T @ A_w
            Atb = A.T @ b_w
            rank = np.linalg.matrix_rank(ATA)
            if rank == 2:
                v = np.linalg.solve(ATA, Atb)
                V1[i, j] = v[0]
                V2[i, j] = v[1]
                Lambda[i, j] = np.min(np.linalg.eigvals(ATA))
            else:
                V1[i, j] = 0
                V2[i, j] = 0
                Lambda[i, j] = 0

    return V1, V2, Lambda

def create_video_vector(image_folder, output, fonction, video='video.mp4', h=3, p=0.5):

    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")])
    if not images:
        print("Aucunne image trouvée dans le dossier.")
        return
    frame = cv2.imread(os.path.join(image_folder, images[0]), cv2.IMREAD_GRAYSCALE)
    height, width = frame.shape

    video_path = os.path.join(output, video)
    video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

    for i in range(len(images) - 1):
        F_r = cv2.imread(os.path.join(image_folder, images[i]), cv2.IMREAD_GRAYSCALE).astype(float)
        F_r1 = cv2.imread(os.path.join(image_folder, images[i + 1]), cv2.IMREAD_GRAYSCALE).astype(float)
        V = fonction(F_r, F_r1)

        result_frame = np.copy(F_r).astype(np.uint8)
        result_frame = cv2.cvtColor(result_frame, cv2.COLOR_GRAY2BGR)

        for k in range(0, height, 10):
            for j in range(0, width, 10):
                u, v = int(V[0][k, j]), int(V[1][k, j])
                if np.sqrt(u ** 2 + v ** 2) > 1:
                    cv2.arrowedLine(result_frame, (j, k), (j + v, k + u), (0, 0, 255), 1, tipLength=0.5)

        video_writer.write(result_frame)
        cv2.imwrite(f"{output}/frame_{i}.png", result_frame)

    video_writer.release()
    print(f"Vidéo de flux optique créée : {output}/{video}")

@jit(nopython=True, parallel=True)
def laplacian(V):

    m, n = V.shape
    lap = np.zeros((m, n))

    for i in prange(1, m - 1):
        for j in prange(1, n - 1):
            lap[i, j] = 0.25 * (V[i + 1, j] + V[i - 1, j] + V[i, j + 1] + V[i, j - 1])

    return lap


@jit(nopython=True, parallel=True)
def var_flux_calc(F_r, F_r1, lambda_=1., epsilon=10e-8, nbitermax=1000):

    m, n = F_r.shape
    V1 = np.zeros((m, n))
    V2 = np.zeros((m, n))

    for k in range(nbitermax):
        V1_old = V1.copy()
        V2_old = V2.copy()

        v1 = laplacian(V1_old)
        v2 = laplacian(V2_old)

        for i in prange(1, m - 1):
            for j in prange(1, n - 1):
                dx, dy, dt = approx_f(F_r, F_r1, i, j)
                b = np.array([dx, dy])
                norm_b_squared = np.dot(b, b)

                if norm_b_squared != 0:
                    v = np.array([v1[i, j], v2[i, j]])
                    alpha = dt
                    v_new = v - ((b.T @ v + alpha) / (lambda_ + norm_b_squared)) * b

                    V1[i, j] = v_new[0]
                    V2[i, j] = v_new[1]

        error = np.max(np.sqrt((V1 - V1_old) ** 2 + (V2 - V2_old) ** 2))
        if error < epsilon:
            break

    return V1, V2

@jit(nopython=True)
def approx_f_v2(F_r, F_r1, i, j):

    m, n = F_r.shape

    if i == 0:
        i_min, i_max = 0, 1
    elif i == m - 1:
        i_min, i_max = m - 2, m - 1
    else:
        i_min, i_max = i - 1, i + 1

    if j == 0:
        j_min, j_max = 0, 1
    elif j == n - 1:
        j_min, j_max = n - 2, n - 1
    else:
        j_min, j_max = j - 1, j + 1

    df_dx = (F_r[i_max, j] - F_r[i, j] + F_r[i_max, j_max] - F_r[i, j_max] + F_r1[i_max, j] - F_r1[i, j] + F_r1[
        i_max, j_max] - F_r1[i, j_max]) / 4
    df_dy = (F_r[i, j_max] - F_r[i, j] + F_r[i_max, j_max] - F_r[i_max, j] + F_r1[i, j_max] - F_r1[i, j] + F_r1[
        i_max, j_max] - F_r1[i_max, j]) / 4
    df_dt = (F_r1[i, j] - F_r[i, j] + F_r1[i_max, j] - F_r[i_max, j] + F_r1[i, j_max] - F_r[i, j_max] + F_r1[
        i_max, j_max] - F_r[i_max, j_max]) / 4

    return df_dx, df_dy, df_dt


@jit(nopython=True, parallel=True)
def laplacian_v2(V):

    m, n = V.shape
    lap = np.zeros((m, n))

    for i in range(1, m - 1):
        for j in range(1, n - 1):
            lap[i, j] = 1 / 3 * (0.5 * (V[i - 1, j] + V[i, j + 1] + V[i + 1, j] + V[i, j - 1]) + 0.25 * (
                        V[i - 1, j - 1] + V[i - 1, j + 1] + V[i + 1, j + 1] + V[i + 1, j - 1]))

    return lap


@jit(nopython=True, parallel=True)
def var_flux_calc_v2(F_r, F_r1, lambda_=10.0, epsilon=10e-8, nbitermax=1000):

    m, n = F_r.shape
    V1 = np.zeros((m, n))
    V2 = np.zeros((m, n))

    for k in range(nbitermax):
        V1_old = V1.copy()
        V2_old = V2.copy()

        v1 = laplacian_v2(V1_old)
        v2 = laplacian_v2(V2_old)

        for i in prange(1, m - 1):
            for j in prange(1, n - 1):
                dx, dy, dt = approx_f_v2(F_r, F_r1, i, j)

                b = np.array([dx, dy])
                norm_b_squared = np.dot(b, b)

                if norm_b_squared != 0:
                    v = np.array([v1[i, j], v2[i, j]])
                    alpha = dt
                    v_new = v - ((b.T @ v + alpha) / (lambda_ + norm_b_squared)) * b

                    V1[i, j] = v_new[0]
                    V2[i, j] = v_new[1]

        error = np.max(np.sqrt((V1 - V1_old) ** 2 + (V2 - V2_old) ** 2))
        if error < epsilon:
            break

    return V1, V2

def compute_norm_matrix(V1, V2):

    return np.sqrt(V1 ** 2 + V2 ** 2)


def create_mask(norm_matrix, threshold_factor=2):

    threshold = np.mean(norm_matrix) * threshold_factor
    mask = norm_matrix > threshold
    return mask.astype(np.uint8)


def post_process_mask(mask):

    kernel = np.ones((20, 20), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def apply_mask(image, mask):

    return cv2.bitwise_and(image, image, mask=mask)


def segment_image(image1, image2):

    V1_var, V2_var = var_flux_calc_v2(image1.astype(float), image2.astype(float))
    norm_matrix_var = compute_norm_matrix(V1_var, V2_var)
    mask_var = create_mask(norm_matrix_var)
    # mask_var = post_process_mask(mask_var)
    segmented_image_var = apply_mask(image1.astype(np.uint8), mask_var)
    return segmented_image_var, V1_var, V2_var, mask_var


def display_hadamard_product_results(image, mask):

    F1 = image * mask
    F2 = image * (1 - mask)

    plt.figure(figsize=(6, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.imshow(F1, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.imshow(F2, cmap='gray')

    plt.show()


def create_video_mask(image_folder, output, video='video.mp4'):

    images = sorted([img for img in os.listdir(image_folder)])
    if not images:
        print("Aucune image trouvée dans le dossier.")
        return

    frame = cv2.imread(os.path.join(image_folder, images[0]), cv2.IMREAD_GRAYSCALE)
    height, width = frame.shape

    video_path = os.path.join(output, video)
    video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

    for i in prange(len(images) - 1):
        F_r = cv2.imread(os.path.join(image_folder, images[i]), cv2.IMREAD_GRAYSCALE).astype(float)
        F_r1 = cv2.imread(os.path.join(image_folder, images[i + 1]), cv2.IMREAD_GRAYSCALE).astype(float)
        segmented_image_var, V1_var, V2_var, mask_var = segment_image(F_r, F_r1)
        display_hadamard_product_results(F_r, mask_var)
        result_frame = cv2.cvtColor(segmented_image_var, cv2.COLOR_GRAY2BGR)
        video_writer.write(result_frame)
        cv2.imwrite(f"{output}/frame{i}.png", result_frame)
    video_writer.release()
    print(f"Vidéo de flux optique créée : {output}/{video}")

def track_point(image_folder, output, video, flux_optique_func):

    images = sorted([img for img in os.listdir(image_folder) if img.endswith('.png') or img.endswith('.jpg')])
    if not images:
        print("Aucune image trouvée dans le dossier.")
        return

    first_image_path = os.path.join(image_folder, images[0])
    first_image = cv2.imread(first_image_path, cv2.IMREAD_GRAYSCALE)
    height, width = first_image.shape

    video_path = os.path.join(output, video)
    video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

    x, y = int(width / 2), int(height / 2)

    for i in range(len(images) - 1):
        img1_path = os.path.join(image_folder, images[i])
        img2_path = os.path.join(image_folder, images[i + 1])

        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

        V = flux_optique_func(img1, img2)

        x += int(V[0][y, x])
        y += int(V[1][y, x])

        colored_frame = cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.circle(colored_frame, (x, y), 2, (0, 255, 0), -1)

        video_writer.write(colored_frame)

    video_writer.release()
    print(f"Video sauvegarder: {video_path}")
