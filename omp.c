#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define MAX_MATRIX 1000

int N = MAX_MATRIX;
int DEBUG = 0;
float A[MAX_MATRIX][MAX_MATRIX], B[MAX_MATRIX], X[MAX_MATRIX];

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

void forwardElimination();

void backwardElimination();

void generateMatrix();

int checkEquation();

int main() {
    double starts=omp_get_wtime();

    generateMatrix();
    forwardElimination();
    backwardElimination();

    int matrixStatus = checkEquation();
    double ends=omp_get_wtime();
    double duration = ends - starts;

    if (matrixStatus == 0){
        printf("The eliminated matrix is not consistent.\n");
    }
    else if (matrixStatus < 0) {
        printf("The eliminated matrix has infinity result\n");
    }
    else {
        printf("Finishing elimination.\n");
    }

    printf("------------------------------------\n");
    printf("Elapsed time: %lf s.\n", duration);
    printf("------------------------------------\n");
}

int checkEquation() {
    int i, j;
    for (i = 0; i < N; i++) {
        float consistFlag = 0;
        for (j = 0; j < N; j++) {
            if (A[i][j] != 0.0) consistFlag = 1;
        }
        if (consistFlag == 0 && B[i] != 0) {
            return 0;
        }
        if (consistFlag == 0 && B[i] == 0){
            return -1;
        }
    }
    return 1;
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
