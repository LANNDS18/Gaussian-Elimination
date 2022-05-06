#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int N = 10;

int generate_matrix(float A[N][N], float B[N]) {
    int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            A[i][j] = rand() % 10;
        }
        B[i] = rand() % 10;
    }

    // show matrix
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            printf("%f ", A[i][j]);
        }
        printf("\n");
    }
    printf("\n");
    for (i = 0; i < N; i++) {
        printf("%f ", B[i]);
    }
    printf("\n");
    return 0;
}


int main() {
    float random_matrix[N][N];
    float B[N], X[N];
    float ratio;
    int i, j, k;
    generate_matrix(random_matrix, B);

    for (i = 0; i < N - 1; i++) {
        #pragma omp parallel for num_threads(10) schedule(runtime) private(j, k, ratio)
        for (j = i + 1; j < N; j++) {
            ratio = random_matrix[j][i] / random_matrix[i][i];
            for (k = i; k < N; k++) {
                random_matrix[j][k] -= ratio * random_matrix[i][k];
            }
            B[j] -= ratio * B[i];
            printf("cpu: %d\n", omp_get_thread_num());
        }
    }

    // backwards substitution
    for (i = N - 1; i >= 0; i--) {
        X[i] = random_matrix[i][N];
        for (j = i + 1; j < N; j++) {
            X[i] -= random_matrix[i][j] * X[j];
        }
        X[i] /= random_matrix[i][i];
    }
    // print result
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            printf("%f\t", random_matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
    for (i = 0; i < N; i++) {
        printf("%f\t", B[i]);
    }

}
