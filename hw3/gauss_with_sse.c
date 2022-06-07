#include<math.h>
#include<time.h>
#include <pmmintrin.h>
#include <xmmintrin.h>
#include"mpi.h"
#include"stdio.h"
#include<stdlib.h>
const int rows = 8; /*the rows of matrix*/
const int cols = 8; /*the cols of matrix*/
int main(int argc, char **argv)

{
	int i,j,k,myid,numprocs,anstag;
	double  A[rows][cols],B[cols],X[rows],AB[rows][cols+1];
	int masterpro;
	double buf[cols+1];
	double starttime,endtime;
	double tmp,totaltime;
	srand((unsigned int)time(NULL));
	MPI_Status status;
	masterpro = 0;
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&myid);
	MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
	if (myid==0)
		{
		  starttime = MPI_Wtime();
		  for(i=0;i<cols;i++) {
				B[i]=rand()%20;
				for(j=0;j<rows;j++) {
					A[j][i]=rand()%20;
				}
		  }
		  printf("B矩阵：\n");
		  for(i=0;i<cols;i++)
			  {
				 printf("%.2f\n",B[i]);
			  }
		  printf("A矩阵：\n");
		  for(i=0;i<rows;i++)
			  {
				 for(j=0;j<cols;j++)
				 printf("%.2f ",A[i][j]);
				 printf("\n");

			   }
		  printf("AB矩阵：\n");
		  for (i=0;i<rows;i++)
			  {

				 AB[i][cols]=B[i];
				 for(j=0;j<cols;j++)
				 AB[i][j]=A[i][j];
			   }
		  for(i=0;i<rows;i++)
			  {
				  for(j=0;j<cols+1;j++)
				  printf("%.2f ",AB[i][j]);
				  printf("\n");

			   }


	   }
	int x;
	double coe;
	for(x=0;x<rows;x++)
	   {
		   MPI_Bcast(&AB[x][0],cols+1,MPI_DOUBLE,0,MPI_COMM_WORLD);
		   if(myid==0)
				{
				  for(i=1;i<numprocs;i++)
					{
					  for(k=x+1+i;k<rows;k+=numprocs)
					  MPI_Send(&AB[k][0],cols+1,MPI_DOUBLE,i,0,MPI_COMM_WORLD);
					}
				   for(k=x+1;k<rows;k+=numprocs)
					 {
					   coe=AB[k][x]/AB[x][x];
					   for(j=x;j<cols+1;j++)
						  {
						   AB[k][j]-=AB[x][j]*coe;
						  }
					  }
				   for(i=1;i<numprocs;i++)
					  {
						for(k=x+1+i;k<rows;k+=numprocs)
						{
						 MPI_Recv(&AB[k][0],cols+1,MPI_DOUBLE,i,1,MPI_COMM_WORLD,&status);
						}
					  }
				}
		   else
			   {
                  __m128 t1, t2, t3, t4;
				  for(k=x+1+myid;k<rows;k+=numprocs) {
						MPI_Recv(&AB[k-1][0],cols+1,MPI_DOUBLE,0,0,MPI_COMM_WORLD,&status);

						coe=AB[k-1][x]/AB[x][x];
                        float tmp[4] = { A[k][x], A[k][x], A[k][x], A[k][x] };
                        t1 = _mm_loadu_ps(tmp);
                        for (int j = rows - 4; j > k;j -= 4 * numprocs) {
                            t2 = _mm_loadu_ps(AB[k] + j);
                            t3 = _mm_loadu_ps(AB[x] + j);
                            t4 = _mm_sub_ps(t2,_mm_mul_ps(t1, t3)); 
                            _mm_storeu_ps(AB[k] + j, t4);

                        }
                        for (int j = x + 1; j % 4 !=(rows % 4); j++) {
                            A[k][j] = A[k][j] - A[k][x] * A[x][j];
                        }
						// for(j=x;j<cols+1;j++) {
                        //     AB[k-1][j]-=AB[x][j]*coe;
                        // }
						MPI_Send(&AB[k-1][0],cols+1,MPI_DOUBLE,0,1,MPI_COMM_WORLD);
					  }

			   }


		}
	int result;
	if(myid==0)//判断解得情况
	   {
		  if(fabs(AB[rows-1][cols-1])<0.00001&&fabs(AB[rows-1][cols]<0.00001))//无穷多解
			 {
			   printf("方程组有无穷多解\n");
			   result=1;
			 }
		  if(fabs(AB[rows-1][cols-1])<0.00001&&fabs(AB[rows-1][cols]>0.00001))//无解
			{
			   printf("方程组无解\n");
			   result=-1;
			}
		  else
			{
			  printf("方程组有唯一解\n");
			  result=0;
			}
		}
	double temp=0.0;
	if(result==0)//唯一解，回代解方程
	 {
		if(myid==0)
		 {
			 X[rows-1]=AB[rows-1][cols]/AB[rows-1][cols-1];
			 for(k=rows-2;k>=0;k--)
			   {
				  for (j=k+1;j<cols;j++)
				  temp=temp+AB[k][j]*X[j];
				  X[k]=(AB[k][cols]-temp)/AB[k][k];
			   }
			 printf("方程的解：\n");
			 for(i=0;i<rows;i++)
			 printf("X[%d]=%.2f\n",i,X[i]);

		  }
	 }
	if(result==1)//无穷多解，回代解方程
	 {
		if(myid==0)
		 {
			 X[rows-1]=0;
			 for(k=rows-2;k>=0;k--)
			   {
				  for (j=k+1;j<cols;j++)
				  temp=temp+AB[k][j]*X[j];
				  X[k]=(AB[k][cols]-temp)/AB[k][k];
			   }
			 printf("方程的解：\n");
			 for(i=0;i<rows;i++)
			 printf("X[%d]=%.2f\n",i,X[i]);

		  }
	 }


	endtime = MPI_Wtime();
	totaltime= endtime - starttime;
	if (myid == masterpro)
	printf("total time :%f s.\n",totaltime);
	MPI_Finalize();
	return 0;
}


