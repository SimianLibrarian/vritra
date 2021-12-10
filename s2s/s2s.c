// program compile and run using linux and gcc compiler
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <fstream>
using namespace std;
#define MAXLINE 1000 //every line with end with \n
#define WIDTH  3
 
float distance(int x0,int y0,int z0,int x1,int y1,int z1){
 float d = pow(pow(x0-x1,2)+pow(y0-y1,2)+pow(z0-z1,2),0.5);
 return d;
}

int n_line(std::string fname){
    int number_of_lines = 0;
    std::string line;
    std::ifstream myfile(fname);
    while (std::getline(myfile, line))
        ++number_of_lines;
return number_of_lines;
	}


int main(void){
  FILE *myfile;
  double myvariable;
  int i,j,nl1,nl2;
  nl1=n_line("./first_body.out");
  int x1[nl1];
  int y1[nl1];
  int z1[nl1];
  myfile=fopen("./first_body.out", "r");
  for(i = 0; i < nl1; i++){
    for (j = 0 ; j < WIDTH; j++){
      fscanf(myfile,"%lf",&myvariable);
     if(j==0){x1[i]=myvariable;}
     else if(j==1){y1[i]=myvariable;}
     else if(j==2){z1[i]=myvariable;}
    }
  }
  fclose(myfile);
  nl2=n_line("./second_body.out");
  int x2[nl2];
  int y2[nl2];
  int z2[nl2];
  myfile=fopen("./second_body.out", "r");
  for(i = 0; i < nl2; i++){
    for (j = 0 ; j < WIDTH; j++){
      fscanf(myfile,"%lf",&myvariable);
     if(j==0){x2[i]=myvariable;}
      else if(j==1){y2[i]=myvariable;}
      else if(j==2){z2[i]=myvariable;}
    }
  }
  fclose(myfile);

for(i=0;i<MAXLINE;i++){
 for(j=0;j<MAXLINE;j++){
  printf("%f\n",distance(x1[i],y1[i],z1[j],x2[j],y2[j],z2[j]));}
	}

return 0;
}
