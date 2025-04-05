#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>
#include <cmath>
#include <iomanip>
using namespace std;

#define time 86400
#define AU_TO_M 1.496e+11
#define M_TO_AU 1.0/1.496e+11
#define AU_PER_DAY_TO_MS 1.731e+6
#define MS_TO_AU_PER_DAY 1.0/1.731e+6
#define G 6.67430e-11
#define N 9
#define DAY 1000

__global__ void compute_gravitational_force(double *old_pos, double *old_vel, double *mass, double *new_pos, double *new_vel)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>= N || i==0) return; 

    double ax = 0.0, ay = 0.0, az = 0.0;
    double new_ax = 0.0, new_ay = 0.0, new_az = 0.0;

    
    for (int j=0;j<N;j++) {
        if (i!=j) {
            
            double dx=old_pos[j*3]-old_pos[i*3];
            double dy=old_pos[j*3+1]-old_pos[i*3+1];
            double dz=old_pos[j*3+2]-old_pos[i*3+2];
            double squared_distance=fmax(dx*dx+dy*dy+dz*dz, 1e-9);
            double distance=sqrt(squared_distance);
            double force=(G*mass[j])/(distance*squared_distance);

            ax+=force*dx;
            ay+=force*dy;
            az+=force*dz;
        }
    }

   
    double vx_1=old_vel[i*3]+0.5*ax*time;
    double vy_1=old_vel[i*3+1]+0.5*ay*time;
    double vz_1=old_vel[i*3+2]+0.5*az*time;

    new_pos[i*3]=old_pos[i*3]+vx_1*time;
    new_pos[i*3+1]=old_pos[i*3+1]+vy_1*time;
    new_pos[i*3+2]=old_pos[i*3+2]+vz_1*time;

    for (int j=0;j<N;j++) {
        if (i!= j) {

            double dx=new_pos[j*3]-new_pos[i*3];
            double dy=new_pos[j*3+1]-new_pos[i*3+1];
            double dz=new_pos[j*3+2]-new_pos[i*3+2];
            double squared_distance=fmax(dx*dx+dy*dy+dz*dz,1e-9);
            double distance=sqrt(squared_distance);            
            double force=(G*mass[j])/(distance*squared_distance);

            new_ax+=force*dx;
            new_ay+=force*dy;
            new_az+=force*dz;
        }
    }
    
    new_vel[i*3]=vx_1+0.5*new_ax*time;
    new_vel[i*3+1]=vy_1+0.5*new_ay*time;
    new_vel[i*3+2]=vz_1+0.5*new_az*time;
}


vector<vector<double>> reading_csv(const string &file_name) 
{
    ifstream file(file_name);
    vector<vector<double>> data;
    string line;
    getline(file, line);
    
    while (getline(file,line)) 
    {
        stringstream ss(line);
        vector<double> row;
        string value;
        getline(ss, value, ',');
        while (getline(ss, value, ',')) 
        {
            row.push_back(stod(value)); 
            
        }

        data.push_back(row);
    }
   
    file.close();
    return data;
}


void flatten_data(vector<vector<double>> &planet_data,double *pos, double *vel , double *mass)
{
    for(int i=0;i<planet_data.size();i++){
        mass[i]=planet_data[i][0]; 
        pos[i*3]=planet_data[i][1]*AU_TO_M;  
        pos[i*3+1]=planet_data[i][2]*AU_TO_M; 
        pos[i*3+2]=planet_data[i][3]*AU_TO_M;
        vel[i*3]=planet_data[i][4]*AU_PER_DAY_TO_MS;  
        vel[i*3+1]=planet_data[i][5]*AU_PER_DAY_TO_MS;  
        vel[i*3+2]=planet_data[i][6]*AU_PER_DAY_TO_MS;  
         
    }
}

double* to_cuda_memory(double *host_data,int n){
    double *device_data;
    cudaMalloc(&device_data,n*sizeof(double));
    cudaMemcpy(device_data,host_data,n*sizeof(double),cudaMemcpyHostToDevice);
    return device_data;
}

void save_to_csv(int day, double *position, double *velocity) {
    vector<string> planet_names={
        "Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"
    };

    for (int i = 0; i < N; i++) {
        string filename=planet_names[i]+".csv";
        ofstream file;
        file.open(filename, ios::app);        
        if (day == 1) {
            file << "Day,X,Y,Z,VX,VY,VZ\n";
        }

        file<<day<<"," 
             <<position[i*3]*M_TO_AU<<","<<position[i*3+1]*M_TO_AU<<","<<position[i*3+2]*M_TO_AU<<","
             <<velocity[i*3]*MS_TO_AU_PER_DAY<<","<<velocity[i*3+1]*MS_TO_AU_PER_DAY<<","<<velocity[i*3+2]*MS_TO_AU_PER_DAY<<"\n";
        file.close();
    }
}

int main() 
{
    string file_name="solar_system.csv";  
    vector<vector<double>> planet_data = reading_csv(file_name);
    int row=planet_data.size();
    int col=planet_data[0].size();

    double *position=new double[N*3];
    double *velocity=new double[N*3];
    double *mass=new double[N];

    
    flatten_data(planet_data,position,velocity,mass);

    double *position_c=to_cuda_memory(position,N*3);
    double *velocity_c=to_cuda_memory(velocity,N*3);
    double *mass_c=to_cuda_memory(mass,N);

    int block_size=9;
    int grid_size=1;
    double *new_position,*new_velocity;
    cudaMalloc(&new_position,N*3*sizeof(double));
    cudaMalloc(&new_velocity,N*3*sizeof(double));
       

    for (int day = 1; day <= DAY; day++) {
        compute_gravitational_force<<<grid_size, block_size>>>(position_c, velocity_c, mass_c, new_position, new_velocity);
        cudaDeviceSynchronize();

        cudaMemcpy(position, new_position, N * 3 * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(velocity, new_velocity, N * 3 * sizeof(double), cudaMemcpyDeviceToHost);

        save_to_csv(day, position, velocity);

        swap(position_c, new_position);
        swap(velocity_c, new_velocity);
    }
    
    cudaFree(position_c);
    cudaFree(velocity_c);
    cudaFree(mass_c);
    cudaFree(new_position);
    cudaFree(new_velocity);
    return 0;
}
