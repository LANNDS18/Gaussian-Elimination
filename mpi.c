#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#define MAX_MATRIX 1000

int N = MAX_MATRIX;
int DEBUG = 0;
float A[MAX_MATRIX][MAX_MATRIX], B[MAX_MATRIX];
float X[MAX_MATRIX];
int size, rank;
double commTime = 0, computationTime = 0;

void generateMatrix();
void forwardStep(int, int);
void forwardElimination();
void backwardElimination();
void displayMatrices();

int main() {
    double start, finish, loc_elapsed, elapsed, maxCommTime, maxCompTime;

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
    MPI_Reduce(&commTime, &maxCommTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&computationTime, &maxCompTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("----Gaussian Elimination in MPI-----\n");
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
            } else {
                if (i == N - 1) {
                    printf("The eliminated matrix is consistent.\n");
                }
            }
        }
        printf("------------------------------------\n");
        printf("Elapsed time: %lf s.\n", elapsed);
        printf("Communication time: %lf s.\n", maxCommTime);
        printf("Computation time: %lf s.\n", maxCompTime);
        printf("------------------------------------\n");
    }
    displayMatrices();
    MPI_Finalize();
    return 0;
}


void generateMatrix() {
    if (rank == 0) {
        int i, j;
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                A[i][j] = rand() % 30000;
            }
            B[i] = rand() % 30000;
            X[i] = 0.0f;
        }
    }
}


void forwardStep(int pivot, int current) {
    computationTime -= MPI_Wtime();
    float ratio = A[current][pivot] / A[pivot][pivot];
    for (int k = pivot; k < N; k++)
    {
        A[current][k] -= A[pivot][k] * ratio;
    }
    B[current] -= B[pivot] * ratio;
    computationTime += MPI_Wtime();
}


void backwardElimination() {
    if (rank == 0) {
        computationTime -= MPI_Wtime();
        int i, j;
        for (i = N - 1; i >= 0; i--) {
            X[i] = B[i];
            for (j = i + 1; j < N; j++) {
                X[i] -= A[i][j] * X[j];
            }
            X[i] /= A[i][i];
        }
        computationTime -= MPI_Wtime();
    }
}


void forwardElimination(){

    MPI_Status status;
    int process, pivot, row;

    // loop on each row as pivot then send to other processes
    for (pivot = 0; pivot < N - 1; pivot++) {
        // Broadcast the current row to other processes
        commTime -= MPI_Wtime();
        MPI_Bcast(&A[pivot], N, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&B[pivot], 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
        commTime += MPI_Wtime();

        // Root process as the master process
        if(rank == 0) {
            // Send chunk of rows according to the rank of the process
            for (process = 1; process < size; process++) {
                for (row = pivot + 1 + process; row < N; row += size){
                    commTime -= MPI_Wtime();
                    MPI_Send(&A[row], N, MPI_FLOAT, process, 0, MPI_COMM_WORLD);
                    MPI_Send(&B[row], 1, MPI_FLOAT, process, 0, MPI_COMM_WORLD);
                    commTime += MPI_Wtime();
                }
            }
            // Elimination for root process
            for (row= pivot + 1 ; row < N; row += size) {
                forwardStep(pivot, row);
            }
            // Receive the result from other processes
            for (process = 1; process < size; process++){
                for (row = pivot + 1 + process; row < N; row += size){
                    commTime -= MPI_Wtime();
                    MPI_Recv(&A[row], N, MPI_FLOAT, process, 1, MPI_COMM_WORLD, &status);
                    MPI_Recv(&B[row], 1, MPI_FLOAT, process, 1, MPI_COMM_WORLD, &status);
                    commTime += MPI_Wtime();
                }
            }
        }
        else {
            // loop on the assigned chunk of the matrix (non-root)
            for (row = pivot + 1 + rank; row < N; row += size) {
                commTime -= MPI_Wtime();
                MPI_Recv(&A[row], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
                MPI_Recv(&B[row], 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
                commTime += MPI_Wtime();
                forwardStep(pivot, row);
                commTime -= MPI_Wtime();
                MPI_Send(&A[row], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
                MPI_Send(&B[row], 1, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
                commTime += MPI_Wtime();
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
