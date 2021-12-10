// program compile and run using linux and gcc compiler
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <fstream>
//#include <assert.h>

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

//function to determine if a point r is in the cylinder between pt1 and pt2
/*def points_in_cylinder(pt1, pt2, r, q): r:radius, q:set of points tested
    vec = pt2 - pt1
    const = r * np.linalg.norm(vec)
    condition_1=np.where(np.dot(q-pt1,vec)>=0)
    condition_2=np.where(np.dot(q-pt2,vec)<=0)
    condition_3=np.where([np.linalg.norm(np.cross(x-pt1,vec)) for x in q]<=const)
    return [int(x) for x in range(np.max(condition_1)) if x in condition_1[0] and x in condition_2[0] and x in condition_3[0]]*/



bool points_in_cylinder(float *pt1,float *pt2,float radius,double *T){
	// Vector joining the two centroids
	float R[3],t[3];
	for(int ii=0;ii<3;ii++){R[ii] = pt2[ii]-pt1[ii];/*printf("%f %f \n",pt2[ii]-pt&[ii],R[ii]);*/}
	//printf("Imported voxel position : ");
	for(int ii=0;ii<3;ii++){t[ii]=(double)T[ii];/*printf("%f ",T[ii]);*/}
	//printf("\n");
	/*printf("Intercentroid distance : %f %f %f\n",R[0],R[1],R[2]);
	printf("Voxel position : %f %f %f\n",t[0],t[1],t[2]);*/
	
	float norm = pow(R[0]*R[0]+R[1]*R[1]+R[2]*R[2],0.5); 
	float condition_1 = (t[0]-pt1[0])*R[0]+(t[1]-pt1[1])*R[1]+(t[2]-pt1[2])*R[2]; // must be larger than or equal to 0
	float condition_2 = (t[0]-pt2[0])*R[0]+(t[1]-pt2[1])*R[1]+(t[2]-pt2[2])*R[2]; //must be smaller than or equal to 0
	float condition_3 = pow(pow((t[1]-pt1[1])*R[2]-(t[2]-pt1[2])*R[1],2)+pow((t[2]-pt1[2])*R[0]-(t[0]-pt1[0])*R[2],2)+pow((t[0]-pt1[0])*R[1]-(t[1]-pt1[1])*R[0],2),0.5); //must be smaller than or equal to radius*intercentroid distance

	return (condition_1>=0) and (condition_2<=0) and (condition_3/norm<=radius);
}


int main(int argc, char **argv){

//First step: import the surface voxels of two indicated indices and put them in two separate files
  long int N1=11;
  long int N2=12;
  int qq=0;
long int temp=0;
        while(argc--){
        temp = atol(*argv++);
        if(qq>0){
        if(qq==1){N1 = temp;}
        else if(qq==2){N2 = temp;}
        }
        qq=qq+1;
                }

  
//Importing the centroids from N1 and N2 drops
  FILE *myFile, *fp1, *fp2;
  if ( (myFile = fopen("../SlicesY_Contours/centroids.csv", "r") ) == NULL )     
  {
     cout << "Error: can't open file" << endl;
     exit(1);
  }
  int nlc = n_line("../SlicesY_Contours/centroids.csv");

  float centroid_1[3];
  float centroid_2[3];
  char line[100];
  for(int i = 1; i < nlc; i++){
	fscanf(myFile, "%s", line); 
	//printf("%s\n",line);
	char delim[] = ";";
	char *ptr = strtok(line,delim);

	//Checking if ptr matches the index N1
	if(atoi(ptr)==N1){//printing out the datas only if its with the first index
	int k=0;
	while(ptr != NULL){
	if(k>0){centroid_1[k-1]=atof(ptr);}
	k=k+1;
	ptr = strtok(NULL, delim);}}
	
	//Checking if ptr matches the index N2
	else if(atoi(ptr)==N2){//printing out the datas only if its with the second index
	int k=0;
	while(ptr != NULL){
	if(k>0){centroid_2[k-1]=atof(ptr);}
	k=k+1;ptr = strtok(NULL, delim);
			}
		}	
	}
	fclose(myFile);
	
printf("Centroid %ld : %f %f %f\n",N1,centroid_1[0],centroid_1[1],centroid_1[2]);
printf("Centroid %ld : %f %f %f\n",N2,centroid_2[0],centroid_2[1],centroid_2[2]);

// Example of how to test if a point is inside the cylinder
//bool inner = points_in_cylinder(centroid_1,centroid_2,10.0,centroid_1);
//printf("\nInside cylinder : %d \n",inner);
  
  char myvariable;
  int nl1;
  nl1=n_line("../SlicesY_Contours/contours.csv");
  int x1[100];
  int y1[100];
  int z1[100];
  double pos[3];



//FILE *myFile;
  if ( (myFile = fopen("../SlicesY_Contours/contours.csv", "r") ) == NULL )     
  {
     cout << "Error: can't open file" << endl;
     exit(1);
  }
  
  if ( (fp1 = fopen("./first_body.out", "w") ) == NULL )     
  {
     cout << "Error: can't open file" << endl;
     exit(1);
  }
    if ( (fp2 = fopen("./second_body.out", "w") ) == NULL )     
  {
     cout << "Error: can't open file" << endl;
     exit(1);
  }

  int NN=0,NNN=0,index=0;
  for(int ii = 1; ii < nl1; ii++){//iterating over all the points
	fscanf(myFile, "%s", line); 
	char delim[] = ";";
	char *ptr = strtok(line,delim);
	if(atoi(ptr)==N1||atoi(ptr)==N2){//taking the points only if they have the good index
	int k=0;
	while(ptr != NULL){
	if(k==0){ index = atoi(ptr);}
		if(k>0){
			pos[k-1]=atof(ptr);//attention, il faudra que j'ajoute l'index en première case de l'array
		}
		k=k+1;
	ptr = strtok(NULL, delim);}//printf("\n");
		if(points_in_cylinder(centroid_1,centroid_2,30.0,pos)){
		if(index==N1){fprintf(fp1,"%f %f %f\n",pos[0],pos[1],pos[2]);}
		else if(index==N2){fprintf(fp2,"%f %f %f\n",pos[0],pos[1],pos[2]);}
		NNN=NNN+1;}
		NN=NN+1;
		}	
	}
	printf("Total number of contour voxels : %d \n",NN);
	printf("Number of voxels for the cap : %d \n",NNN);
	fclose(myFile);fclose(fp1);fclose(fp2);




return 0;
}
