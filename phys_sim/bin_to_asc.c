#include<stdio.h>
#include<stdlib.h>
#include<stddef.h>
#include<math.h>
#include</home/cmslab/pankaj/fftw_local/include/fftw3.h>

int main (void){
  int nx, ny;

  FILE *fpread, *fpout;
  char fr[100],fw[100];
  char prefix[20];
  double total;
  char new[100];
  fftw_complex *comp_b, *comp_c;

  fpread = fopen("InputParams", "r");
  fscanf(fpread,"%s%d", fr, &nx);
  fscanf(fpread,"%s%d", fr, &ny);
  fclose(fpread);

  comp_b = (fftw_complex *)fftw_malloc (nx * ny * sizeof (fftw_complex));
  comp_c = (fftw_complex *)fftw_malloc (nx * ny * sizeof (fftw_complex));
  int i, j, initial, final, steps;

  printf("Initial count\n");
  scanf("%d", &initial);
  printf("Final count\n");
  scanf("%d", &final);
  printf("Steps\n");
  scanf("%d", &steps);

  for (int count = initial; count <=final; count += steps){

     sprintf (fr, "comp.%06d", count);
     printf ("%s\n", fr);
     fpread = fopen (fr, "r");
     fread (&comp_b[0][0], sizeof (double), 2 * nx * ny, fpread);
     fread (&comp_c[0][0], sizeof (double), 2 * nx * ny, fpread);
     fclose (fpread);
  
     sprintf (fw, "prof_gp.%06d", count);
     fpout= fopen(fw,"w");
     for (i = 0; i < nx; i++){
       for (j = 0; j < ny; j++){
        fprintf(fpout,"%d\t%d\t%le\t%le\n ", i, j, comp_b[j + i * ny][0], comp_c[j+i*ny][0]);
       } fprintf(fpout,"\n");
     }fclose(fpout);
  }
  

  fftw_free (comp_b);
  fftw_free (comp_c);
  return(0);
}
