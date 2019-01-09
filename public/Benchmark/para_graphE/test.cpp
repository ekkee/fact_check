/**
	* Copyright (c) 2016 LIBBLE team supervised by Dr. Wu-Jun LI at Nanjing University.
	* All Rights Reserved.
	* Licensed under the Apache License, Version 2.0 (the "License");
	* you may not use this file except in compliance with the License.
	* You may obtain a copy of the License at
	*
	* http://www.apache.org/licenses/LICENSE-2.0
	*
	* Unless required by applicable law or agreed to in writing, software
	* distributed under the License is distributed on an "AS IS" BASIS,
	* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
	* See the License for the specific language governing permissions and
	* limitations under the License. */

#include <iostream>
#include <fstream>
#include <random>
#include <iomanip>
#include <thread>
#include <vector>
#include <algorithm>
#include <map>
#include <set>
#include <tuple>
#include <chrono>
#include <dirent.h>
using namespace std;

#include "transbase.hpp"
#include "transe.hpp"
#include "transr.hpp"
#include "transh.hpp"
#include "transd.hpp"
#include "spheree.hpp"
#include "json.hpp"

using json = nlohmann::json;
//parameters of program
int nthreads = 1;
string data_path = "";
int batch_size = 1;

string method = "TransE";
string out_file = "";
string in_file = "";
transbase *trans_ptr;

//hyper-parameters of algorithm
int l1_norm = 1;

//parameters of knowledge_graph
int  test_num = 0,  embedding_dim;
int entity_num = 0, relation_num = 0 ; // initial value
//data structure of algorithm
vector<int> test_h, test_r, test_t, test_label;
vector<string> name_h, name_r, name_t;
vector<double> predict_proba;
map<string, int> entity_dict;

map<string, int> relation_dict;

int arg_handler(string str, int argc, char **argv) {
	int pos;
	for (pos = 0; pos < argc; pos++) {
		if (str.compare(argv[pos]) == 0) {
			return pos;
		}
	}
}

string ReplaceString(string subject, const string& search,
                          const string& replace) {
    size_t pos = 0;
    while ((pos = subject.find(search, pos)) != string::npos) {
         subject.replace(pos, search.length(), replace);
         pos += replace.length();
    }
    return subject;
}

void initial() {
	ifstream file;		
	int id;
	string tmp;
	

	file.open(data_path + "graphs/node_dict.tsv");
	
	while(!file.eof()) {
		file >> id ;
		if( file.eof() ) break;
		file.get(); 
		file >>  tmp ;
		entity_dict[tmp] = id;
		entity_num = entity_num + 1;	
	}
	file.close();

	

	file.open(data_path + "graphs/edge_dict.tsv");
	
	while(!file.eof()) {
		file >> id ;
		if( file.eof() ) break;
		file.get(); 
		file >> tmp ;
		relation_dict[tmp] = id;
		relation_num = relation_num + 1;	
	}
	file.close();

}
	
void load_test_fold(string test_file) {
	ifstream file;		
	test_h.clear();
	test_r.clear();
	test_t.clear();
	name_h.clear();
	name_r.clear();
	name_t.clear();
	test_label.clear();

	file.open(test_file);
	int l;
	string h, r, t, tmp;
	// ignore first line
	file >> h;
	file.get();		
	file >> t;
	file.get(); 
	file >> r;
	file.get();
	file >> l;
	file.get();
	file >> tmp;
	file.get();
	file >> tmp;	
	// start reading
	while(!file.eof()) {
		file >> h;
		if( file.eof() ) break;
		file.get();		
		file >> t;
		file.get(); 
		file >> r;
		file.get();
		file >> l;
		file.get();
		file >> tmp;
		file.get();
		file >> tmp;		
		//	cout << h << " " << t << " " << l << endl;
		test_h.push_back(entity_dict[h]-1);
		name_h.push_back(h);
		test_r.push_back(relation_dict[r]-1);
		name_r.push_back(r);
		test_t.push_back(entity_dict[t]-1);
		name_t.push_back(t);
		test_label.push_back(l);
		test_num = test_num + 1;
	}
	file.close();
}

void method_ptr_binding(string method) {
	if (method.compare("TransE") == 0) 
		trans_ptr = new transE(0, 0, 0, l1_norm, 0, 0);	//the other parameters may be modified in read_from_file
	else if (method.compare("TransR") == 0)
		trans_ptr = new transR(0, 0, 0, 0, l1_norm, 0, 0);
	else if (method.compare("TransH") == 0) 
		trans_ptr = new transH(0, 0, 0, l1_norm, 0, 0, 0);
	else if (method.compare("TransD") == 0)
		trans_ptr = new transD(0, 0, 0, l1_norm, 0, 0);
	else if (method.compare("SphereE") == 0)
		trans_ptr = new sphereE(0, 0, 0, l1_norm, 0, 0, 0);
	else {
		cout << "no such method!" << endl;
		exit(1);
	}
}

int main(int argc, char **argv) {
	//arg processing
	// int pos;
	// if ((pos = arg_handler("-nthreads", argc, argv)) > 0) nthreads = atoi(argv[pos + 1]);
	// if ((pos = arg_handler("-numtest", argc, argv)) > 0) test_num = atoi(argv[pos + 1]);
	// if ((pos = arg_handler("-path", argc, argv)) > 0) data_path = string(argv[pos + 1]);
	// if ((pos = arg_handler("-batch_size", argc, argv)) > 0) batch_size = atoi(argv[pos + 1]);
	// if ((pos = arg_handler("-method", argc, argv)) > 0) method = string(argv[pos + 1]);
	// if ((pos = arg_handler("-output", argc, argv)) > 0) out_file = string(argv[pos + 1]);
	// if ((pos = arg_handler("-input", argc, argv)) > 0) in_file = string(argv[pos + 1]);
	// if ((pos = arg_handler("-l1_norm", argc, argv)) > 0) l1_norm = atoi(argv[pos + 1]); 
	// cout << "arg processing done." << endl;
	
	data_path = string(argv[1]);
	string spec_ = string(argv[2]);
	string ex_str = ".json";

	std::ifstream i(data_path + "experiment_specs/" + string(argv[2]));
	json spec;
	i >> spec;

	method = spec["operation"]["method"];
	embedding_dim = spec["operation"]["features"]["embed_dim"];

	nthreads = spec["operation"]["features"]["nprocs"];
	in_file = spec["split"]["test_file"];

	cout << "args settings: " << endl
		<< "----------" << endl
		<< "method " << method << endl
		<< "norm " << (l1_norm == 0 ? "L2" : "L1") << endl
		<< "thread number " << nthreads << endl
		<< "data path " << data_path << endl
		<< "----------" << endl;
	
	//initializing
	initial();
	
	method_ptr_binding(method);
	
	// trans_ptr->read_from_file(data_path+"graphs/", spec["split"]["name"]);
	trans_ptr->read_from_file(data_path+"graphs/", "InComplete");
	cout << "initializing process done." << endl;
	
	//testing
	ofstream file;
	out_file = ReplaceString(in_file, "scenario", method + "_score");
	// system("mkdir -p " + out_file);
	// file.open( out_file + "/score.tsv");
	file.open( out_file);
	file << "s" << "\t" << "o" << "\t" << "p" << "\t" << "true_label" << "\t" << "predict_proba" << endl;
	load_test_fold(in_file);

	int count = 0;
	while (count < test_num) {
		int h = test_h[count], r = test_r[count], t = test_t[count]; 		
		double loss = trans_ptr->triple_loss(h, r, t);
		predict_proba.push_back(loss);
		count += 1;
	}
	auto min_max = minmax_element(predict_proba.begin(),predict_proba.end());
	double min_loss = *min_max.first;
	double max_loss = *min_max.second;
	count = 0;
	while (count < test_num) {
		cout << count << endl;
		int h = test_h[count], r = test_r[count], t = test_t[count];
	    // cout<< h << " "<< r << " " << t << endl;  		
		file << name_h[count] << "\t" << name_t[count] << "\t" << name_r[count] << "\t" << test_label[count] << "\t" << (1-(predict_proba[count]-min_loss)/(max_loss-min_loss)) << endl;
		count += 1;
	}
	file.close();

   //  DIR *dir = opendir(in_file);
   //  struct dirent *entry = readdir(dir);

   //  while(entry = readdir(dir)) {

   //      if (entry->d_type == DT_DIR && strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0 )
   //          count << " Fold: "  << entry->d_name << endl;
			// ofstream file;
			// out_file = in_file + "/" + entry->d_name + "/" +  spec_.erase(spec_.find(ex_str), ex_str.length()) 
			// system("mkdir -p " + out_file);
			// file.open( out_file + "/score.tsv");
			// file << "s" << "\t" << "o" << "\t" << "p" << "\t" << "true_label" << "\t" << "predict_proba" << endl;
			// load_test_fold(in_file + "/" + entry->d_name + "/testing.tsv")
			// int count = 0;
			// while (count < test_num) {
			// 	cout << count << endl;
			// 	int h = test_h[count], r = test_r[count], t = test_t[count];
			//     // cout<< h << " "<< r << " " << t << endl;  		
			// 	double loss = trans_ptr->triple_loss(h, r, t);
		 //        // cout<<loss<<endl;
			// 	file << name_h[count] << "\t" << name_t[count] << "\t" << name_r[count] << "\t" << test_label[count] << "\t" << loss << endl;
			// 	count += 1;
			// }
			// file.close();

   //  }
   //  closedir(dir);



	cout << "testing process done "  << endl;
	

}
