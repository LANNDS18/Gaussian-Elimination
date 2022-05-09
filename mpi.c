#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

#define MAX_MATRIX 1000

int N = MAX_MATRIX;
int DEBUG = 0;
float A[MAX_MATRIX][MAX_MATRIX], B[MAX_MATRIX];
float X[MAX_MATRIX];
int size, rank;

void generateMatrix();
void forwardStep(int, int);
void forwardElimination();
void backwardElimination();
int checkEquation();
void displayMatrices();


int main() {
    double start, finish, loc_elapsed, elapsed;

    start = MPI_Wtime();

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Barrier(MPI_COMM_WORLD);

    generateMatrix();
    forwardElimination();
    backwardElimination();

    finish = MPI_Wtime();
    loc_elapsed = finish-start;
    MPI_Reduce(&loc_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        int matrixStatus = checkEquation();
        if (matrixStatus == 0){
            printf("The eliminated matrix is not consistent.\n");
        }
        else if (matrixStatus < 0) {
            printf("The eliminated matrix has infinity result\n");
        }
        else {
            printf("Finishing elimination.\n");
        }
        printf("Elapsed time: %lf s.\n", elapsed);
    }

    displayMatrices();
    MPI_Finalize();
    return 0;
}


void generateMatrix() {
    // float K[MAX_MATRIX][MAX_MATRIX] = {{1,21,3}, {4,5,6}, {7,8,9}};
    // float Y[MAX_MATRIX] = {1, 3, 1};
    if (rank == 0) {
        int i, j;
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                A[i][j] = rand() % 10;
            }
            B[i] = rand() % 10;
            X[i] = 0.0f;
        }
        printf("\nStarting clock.\n");
    }
}


void forwardStep(int pivot, int current) {
    float ratio = A[current][pivot] / A[pivot][pivot];
    for (int k = pivot; k < N; k++)
    {
        A[current][k] -= A[pivot][k] * ratio;
    }
    B[current] -= B[pivot] * ratio;
}


void backwardElimination() {
    if (rank == 0) {
        int i, j;
        for (i = N - 1; i >= 0; i--) {
            X[i] = B[i];
            for (j = i + 1; j < N; j++) {
                X[i] -= A[i][j] * X[j];
            }
            X[i] /= A[i][i];
        }
    }
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

    MPI_Status status;
    int process, norm, j;

    for (norm = 0; norm < N - 1; norm++) {
        // Broadcast the current row to other processes
        MPI_Bcast(&A[norm], N, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&B[norm], 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

        // Root process as the master process
        if(rank == 0) {
            // Send rows according to the rank of the process
            for (process = 1; process < size; process++) {
                for (j= norm + 1 + process; j < N; j+=size){
                    MPI_Send(&A[j], N, MPI_FLOAT, process, 0, MPI_COMM_WORLD);
                    MPI_Send(&B[j], 1, MPI_FLOAT, process, 0, MPI_COMM_WORLD);
                }
            }
            for (j= norm + 1 ; j < N; j += size) {
                forwardStep(norm, j);
            }
            for (process = 1; process < size; process++){
                for (j = norm + 1 + process; j < N; j += size){
                    MPI_Recv(&A[j], N, MPI_FLOAT, process, 1, MPI_COMM_WORLD, &status);
                    MPI_Recv(&B[j], 1, MPI_FLOAT, process, 1, MPI_COMM_WORLD, &status);
                }
            }
        }
        else {
            // loop on the assigned chunk of the matrix
            for (j = norm + 1 + rank; j < N; j += size) {
                MPI_Recv(&A[j], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
                MPI_Recv(&B[j], 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
                forwardStep(norm, j);
                MPI_Send(&A[j], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
                MPI_Send(&B[j], 1, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

void displayMatrices() {
    if (rank == 0 && DEBUG) {
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
}
