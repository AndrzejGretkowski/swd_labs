#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def vectors_uniform(k):
    """Uniformly generates k vectors."""
    vectors = []
    for a in np.linspace(0, 2 * np.pi, k, endpoint=False):
        vectors.append(2 * np.array([np.sin(a), np.cos(a)]))
    return vectors


def visualize_transformation(A, vectors):
    """Plots original and transformed vectors for a given 2x2 transformation matrix A and a list of 2D vectors."""
    for i, v in enumerate(vectors):
        # Plot original vector.
        plt.quiver(0.0, 0.0, v[0], v[1], width=0.008, color="blue", scale_units='xy', angles='xy', scale=1,
                   zorder=4)
        plt.text(v[0]/2 + 0.25, v[1]/2, "v{0}".format(i), color="blue")

        # Plot transformed vector.
        tv = A.dot(v)
        plt.quiver(0.0, 0.0, tv[0], tv[1], width=0.005, color="magenta", scale_units='xy', angles='xy', scale=1,
                   zorder=4)
        plt.text(tv[0] / 2 + 0.25, tv[1] / 2, "v{0}'".format(i), color="magenta")
    plt.xlim([-6, 6])
    plt.ylim([-6, 6])
    plt.margins(0.05)
    # Plot eigenvectors
    plot_eigenvectors(A)
    plt.show()


def visualize_vectors(vectors, color="green"):
    """Plots all vectors in the list."""
    for i, v in enumerate(vectors):
        plt.quiver(0.0, 0.0, v[0], v[1], width=0.006, color=color, scale_units='xy', angles='xy', scale=1,
                   zorder=4)
        plt.text(v[0] / 2 + 0.25, v[1] / 2, "eigv{0}".format(i), color=color)


def plot_eigenvectors(A):
    """Plots all eigenvectors of the given 2x2 matrix A."""
    # TODO: Zad. 4.1. Oblicz wektory własne A. Możesz wykorzystać funkcję np.linalg.eig#
    _, eigvec = np.linalg.eig(A)
    # TODO: Zad. 4.1. Upewnij się poprzez analizę wykresów, że rysowane są poprawne wektory własne (łatwo tu o pomyłkę).
    visualize_vectors(eigvec)


def EVD_decomposition(A):
    # TODO: Zad. 4.2. Uzupełnij funkcję tak by obliczała rozkład EVD zgodnie z zadaniem.
    eigval, eigvec = np.linalg.eig(A)
    K = eigvec
    K_inv = np.linalg.inv(K)
    L = np.diag(eigval)
    print(K_inv, L, K)
    # print(K_inv.dot(L).dot(K), A)
    # assert np.array_equal(K_inv.dot(L).dot(K), A)


def plot_attractors(A, vectors):
    # TODO: Zad. 4.3. Uzupełnij funkcję tak by generowała wykres z atraktorami.
    color_gen = (c for c in plt.get_cmap('tab10').colors)
    eig_mapping = []
    eigval, eigvec = np.linalg.eig(A)

    colors = []
    for vector in vectors:
        vector = normalize(vector)
        vec_attractor = find_attractor(A, vector)
        if np.allclose(vec_attractor, vector):
            colors.append((0, 0, 0))
        else:
            found = False
            for (val, color) in eig_mapping:
                if np.allclose(vec_attractor, val, rtol=0.01):
                    colors.append(color)
                    found = True
            if not found:
                c = next(color_gen)
                eig_mapping.append((vec_attractor, c))
                colors.append(c)

    for vector, color in zip(vectors, colors):
       plot_vector(normalize(vector), color, 0.004, 3)

    for vector, color in eig_mapping:
        plot_vector(normalize(vector), color, 0.004, 8)

    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.margins(0.05)
    plt.grid()

    plt.show()

def find_attractor(A, vector):
    new_v = np.copy(vector)
    for _ in range(1000):
        new_v = normalize(A @ new_v)
    return new_v

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

def plot_vector(vector, color, width, headwidth):
    q = plt.quiver(0.0, 0.0, vector[0], vector[1], width=width, color=color, headwidth=headwidth, scale_units='xy', angles='xy', scale=1,
               zorder=4)

def show_eigen_info(A, vectors):
    EVD_decomposition(A)
    visualize_transformation(A, vectors)
    plot_attractors(A, vectors)


if __name__ == "__main__":
    vectors = vectors_uniform(k=16)

    A = np.array([[2, 0],
                  [0, 2]])
    show_eigen_info(A, vectors)


    A = np.array([[-1, 2],
                  [2, 1]])
    show_eigen_info(A, vectors)


    A = np.array([[3, 1],
                  [0, 2]])
    show_eigen_info(A, vectors)


    A = np.array([[2, -1],
                  [1, 4]])
    show_eigen_info(A, vectors)
