#include <omp.h>
#include <stdio.h>

int main()
{
   #pragma omp parallel num_threads(4)
   {
      printf("hello, world\n");
      printf("hello, class\n");
   }
}
