#include<fstream>
#include<stdlib.h> 
#include<stdio.h>
#include<iostream>
#include<time.h> 
#include<cmath>
#include <stdio.h> 
#include <stdlib.h> 

using namespace std;

int main()
{
  srand(time(0)); 
  ofstream city_coord_list;
  city_coord_list.open("city_coord_data.txt");

  int index;
  double x, y;
  ifstream fin; 
  fin.open("bier127.txt");
  //fin.open("kroA100.txt");
  
  fin>>index;
  while(!fin.eof())
  {
    fin>>x;
    fin>>y;
    city_coord_list<<x<<" "<<y<<endl;
    fin>>index;
  }
  std::cout<<"Cleaned the data"<<endl;
  city_coord_list.close();
  return 0;
}
