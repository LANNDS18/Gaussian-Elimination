#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define MAX_MATRIX 1000

int N = MAX_MATRIX;
int DEBUG = 0;
float A[MAX_MATRIX][MAX_MATRIX], B[MAX_MATRIX], X[MAX_MATRIX];

void displayMatrices();
void forwardElimination();
void backwardElimination();
void generateMatrix();

int main() {
    double starts=omp_get_wtime();

    generateMatrix();
    forwardElimination();
    backwardElimination();

    double ends=omp_get_wtime();
    double elapsed = ends - starts;

    printf("----Gaussian Elimination in OMP-----\n");
    int i, j;
    for (i = 0; i < N; i++) {
        float consistFlag = 0;
        for (j = 0; j < N; j++) {
            if (A[i][j] != 0.0) consistFlag = 1;
        }
        if (consistFlag == 0 && B[i] != 0) {
            printf("The eliminated matrix is not consistent.\n");
            break;
        }
        else if (consistFlag == 0 && B[i] == 0){
            printf("The eliminated matrix has infinity result\n");
            break;
        }
        else {
            if (i == N - 1) {
                printf("The eliminated matrix is consistent.\n");
            }
        }
    }
    printf("------------------------------------\n");
    printf("Elapsed time: %lf s.\n", elapsed);
    printf("------------------------------------\n");
}


void forwardElimination(){
    float ratio;
    int i, j, k;
    /* Current Row for computing ratio*/
    for (i = 0; i < N - 1; i++) {
    #pragma omp parallel for num_threads(8) schedule(guided) private(j, k, ratio)
        for (j = i + 1; j < N; j++) {
            ratio = A[j][i] / A[i][i];
            /* Columns of j are scaled by ratio */
            for (k = i; k < N; k++) {
                A[j][k] -= ratio * A[i][k];
            }
            B[j] -= ratio * B[i];
            // printf("cpu: %d\n", omp_get_thread_num());
        }
    }
}

void backwardElimination() {
    int i, j;
    for (i = N - 1; i >= 0; i--) {
        X[i] = B[i];
        for (j = i + 1; j < N; j++) {
            X[i] -= A[i][j] * X[j];
        }
        X[i] /= A[i][i];
    }
}

void generateMatrix() {
    int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            A[i][j] = rand() % 800;
        }
        B[i] = rand() % 10;
    }
}

void displayMatrices() {
    printf("------------------------------------\n");
    printf("Show Matrix: A\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", A[i][j]);
        }
        printf("\n");
    }
    printf("------------------------------------\n");
    printf("Show Matrix: B\n");
    for (int i = 0; i < N; i++) {
        printf("%f\t", B[i]);
    }
    printf("\n------------------------------------\n");
    printf("Show Result: X\n");
    for (int i = 0; i < N; i++) {
        printf("%f\t", X[i]);
    }
    printf("\n");
}
