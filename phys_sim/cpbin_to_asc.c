#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<fftw3.h>

int main (void)
{
 FILE *fpread, *fp;
 char fr[100];
 char junk[100];
 double *comp_a, sum = 0.0, mean;
 double total_b, total_c;
 int initcount;
 fftw_complex *comp_b, *comp_c;
 int index, start, steps, finish, count, ind_final, ind_count;
 int nx, ny;
 
 
 fp = fopen("PostProc.in","r");
 fscanf(fp,"%s%d",junk,&nx); 
 fscanf(fp,"%s%d",junk,&ny); 
 fscanf(fp,"%s%d",junk,&start); 
 fscanf(fp,"%s%d",junk,&steps); 
 fscanf(fp,"%s%d",junk,&finish);
 fscanf(fp,"%s%d",junk,&index);
 fscanf(fp,"%s%d",junk,&ind_final);
 fclose(fp); 
 
 comp_b = (fftw_complex *) fftw_malloc (nx * ny * sizeof (fftw_complex));
 comp_c = (fftw_complex *) fftw_malloc (nx * ny * sizeof (fftw_complex));
 comp_a = (double *) malloc(nx*ny*sizeof(double));

 int i, j;
 for(ind_count=index; ind_count <= ind_final; ind_count++){
	 printf("%d", ind_count);
 for(count=start; count <= finish; count = count + steps){
 
    sprintf (fr, "comp.%06d", count);
    fpread = fopen (fr, "r");
    fread (&comp_b[0][0], sizeof (double), 2 * nx * ny, fpread);
    fread (&comp_c[0][0], sizeof (double), 2 * nx * ny, fpread);
    fclose (fpread);
 
    sprintf (fr, "prof_gp.%06d", count);
    fp = fopen (fr, "w");
    for (i = 0; i < nx; i++) {
     for (j = 0; j < ny; j++) {
      comp_a[j + i * ny] = 1.0 - comp_b[j + i * ny][0] - comp_c[j + i * ny][0];
      fprintf(fp, " %lf ",  comp_a[j + i * ny]);
      }
      fprintf (fp, "\n");
    }
      fprintf (fp, "\n");
    for (i = 0; i < nx; i++) {
     for (j = 0; j < ny; j++) {
      fprintf (fp, " %lf ", comp_b[j + i * ny][0]);
      }
      fprintf (fp, "\n");
    }
      fprintf (fp, "\n");
    for (i = 0; i < nx; i++) {
     for (j = 0; j < ny; j++) {
      fprintf (fp, " %lf ", comp_c[j + i * ny][0]);
     }
      fprintf (fp, "\n");
    }
    fclose(fp);
 }
 }
 fftw_free (comp_b);
 fftw_free (comp_c);
      free (comp_a);

}
