#include<stdio.h>
#include<math.h>

int main(){

int T,n;
int L;
printf("Enter the values of T(thickness), n(no. of folds)");
scanf("%d%d",&T,&n);

L = (3.14/6) * T * (pow(2,n) + 4) * (pow(2,n) - 1);

printf("Minimum Length of the paper is %d.",L);
}