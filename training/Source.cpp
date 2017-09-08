#include "iostream"
#include "opencv.hpp"
#include "fstream"
#include "data_LBP.h"

using namespace std;
using namespace cv;

//define
#define pos_number 484
#define neg_number 1048
#define iteration 10000
#define dimensions 10000
//global variable
data_LBP number[pos_number+neg_number][dimensions];
data_LBP buffer;

float T_plus = 0;
float T_minus = 0;
float S_plus = 0;
float S_minus = 0;

float current_classifier = 0;
int current_classifier_p = 0;
float current_classifier_error = 0;
float current_classifier_error1 = 0;
float current_classifier_error2 = 0;

vector<float> weak_classifier;
vector<int> weak_classifier_dimensions;
vector<float> confidence_value;
vector<int> p;

float current_best_classifier = 0;
int current_best_classifier_index = 0;
float current_best_classifier_error = 1;	//誤差初始化為1  (一開始為1  疊代才拿開始挑最好的弱分類器)
int current_best_classifier_p = 0;

float z = {0};
float weightt[pos_number + neg_number]={0};

int main() {
	//讀取檔案
	fstream pin;
	fstream nin;
	fstream fout;
	pin.open("C:\\Users\\j4355\\Desktop\\pos\\pos_n\\pos.txt",ios::in);
	nin.open("C:\\Users\\j4355\\Desktop\\neg\\LBP\\neg.txt",ios::in);
	fout.open("weak list_5.txt",ios::out);

	if (!pin) {
		std::cout<<"cannot find the postive sample files"<<endl;
		return 0;
	}
	if (!nin) {
		std::cout << "cannot find the negative sample files" << endl;
		return 0;
	}
	//讀取樣本
	for(int pos = 0; pos<pos_number;pos++){
		for(int pos_ = 0; pos_<dimensions;pos_++){
			pin>>number[pos][pos_].data;
			number[pos][pos_].pos_neg = 1;
			weightt[pos] = 1.0 / (2.0*pos_number);
			number[pos][pos_].weight = &weightt[pos]; //initialize the smaple weight
		}
	}
	for (int neg = 0;neg < neg_number;neg++) {
		for (int neg_ = 0; neg_ < dimensions;neg_++) {
			nin>>number[neg +pos_number][neg_].data;
			number[neg + pos_number][neg_].pos_neg = -1;
			weightt[neg + pos_number] = 1.0 / (2.0*neg_number);
			number[neg + pos_number][neg_].weight = &weightt[neg + pos_number];
		}
	}
	//分類
	
	for (int sort_dimen = 0; sort_dimen < dimensions;sort_dimen++) {
		cout<<sort_dimen<<endl;
		for (int sort_min = 0; sort_min < pos_number + neg_number;sort_min++) {
			for (int sort_any = sort_min; sort_any < pos_number + neg_number;sort_any++) {
				if (number[sort_min][sort_dimen].data > number[sort_any][sort_dimen].data) {
					buffer.copy_lan(number[sort_min][sort_dimen]);
					number[sort_min][sort_dimen].copy_lan(number[sort_any][sort_dimen]);
					number[sort_any][sort_dimen].copy_lan(buffer);
				}
			}
		}
	}
	//
	for (int ii = 0; ii < iteration;ii++) {
		std::cout<<"疊代:"<<ii<<endl;
		for (int i_d = 0;i_d < dimensions; i_d++) {
			T_plus = 0;
			T_minus = 0;
			S_plus = 0;
			S_minus = 0;

			current_classifier = 0;
			current_classifier_p = 0;
			current_classifier_error = 0;
			current_classifier_error1 = 0;
			current_classifier_error2 = 0;

			for (int j = 0; j < pos_number + neg_number;j++){
				if (number[j][i_d].pos_neg == 1)
					T_plus	+= *number[j][i_d].weight;
				else 
					T_minus += *number[j][i_d].weight;
			}

			for (int i = 0; i < pos_number + neg_number;i++) {

				if (i == 0) {
					current_classifier			= 0;
					current_classifier_error1	= S_plus  + T_minus - S_minus;
					current_classifier_error2	= S_minus + T_plus  - S_plus;

					current_classifier_error	= (current_classifier_error1<current_classifier_error2)	? current_classifier_error1 : current_classifier_error2;
					current_classifier_p		= (current_classifier_error1<current_classifier_error2) ? -1 : 1;

					if(current_classifier_error < current_best_classifier_error){
						current_best_classifier_error =	current_classifier_error;
						current_best_classifier	= current_classifier;
						current_best_classifier_index =	i_d;
						current_best_classifier_p =	current_classifier_p;
					}
				}
				
				else if(i == (pos_number+neg_number-1)){
					current_classifier = number[i][i_d].data;
					if(number[i][i_d].pos_neg == 1){
						S_plus	+=	*number[i][i_d].weight;
					}
					else{
						S_minus	+=	*number[i][i_d].weight;
					}
					current_classifier_error1 = S_plus	+ T_minus	- S_minus;
					current_classifier_error2 = S_minus + T_plus	- S_plus;

					current_classifier_error	= (current_classifier_error1<current_classifier_error2)	? current_classifier_error1 : current_classifier_error2;
					current_classifier_p		= (current_classifier_error1<current_classifier_error2) ? -1 : 1;
					if (current_classifier_error < current_best_classifier_error) {
						current_best_classifier_error	= current_classifier_error;
						current_best_classifier			= current_classifier;
						current_best_classifier_index	= i_d;
						current_best_classifier_p		= current_classifier_p;
					}
				}

				else{
					current_classifier = (number[i][i_d].data + number[i-1][i_d].data) / 2.0;
					if (number[i][i_d].pos_neg == 1) {
						S_plus += *number[i][i_d].weight;
					}
					else {
						S_minus += *number[i][i_d].weight;
					}
					current_classifier_error1 = S_plus + T_minus - S_minus;
					current_classifier_error2 = S_minus + T_plus - S_plus;

					current_classifier_error	= (current_classifier_error1<current_classifier_error2) ? current_classifier_error1 : current_classifier_error2;
					current_classifier_p		= (current_classifier_error1<current_classifier_error2) ? -1 : 1;

					if (current_classifier_error < current_best_classifier_error) {
						current_best_classifier_error = current_classifier_error;
						current_best_classifier = current_classifier;
						current_best_classifier_index = i_d;
						current_best_classifier_p = current_classifier_p;
					}
				}
			}
		}//dimension
		// 紀錄本輪弱分類器
		p.push_back(current_best_classifier_p);
		weak_classifier.push_back(current_best_classifier);
		weak_classifier_dimensions.push_back(current_best_classifier_index);
		confidence_value.push_back(0.5*log((1 - current_best_classifier_error) / current_best_classifier_error));
		//
		//更新樣本權重	跑1225次
		
			for (int data = 0; data < pos_number + neg_number; data++) {
				int h_of_classifier = 0;
				if (p[ii] == 1) {
					if(number[data][weak_classifier_dimensions[ii]].data > weak_classifier[ii]){
						h_of_classifier =  -1;
					}
					else {
						h_of_classifier =	1;
					}
				}
				else {
					if (number[data][weak_classifier_dimensions[ii]].data > weak_classifier[ii]) {
						h_of_classifier =  1;
					}
					else {
						h_of_classifier = -1;
					}
				}
				*number[data][weak_classifier_dimensions[ii]].weight = *number[data][weak_classifier_dimensions[ii]].weight * exp(-confidence_value[ii] * number[data][weak_classifier_dimensions[ii]].pos_neg * h_of_classifier);	 //更新每個樣本的權重
				z += *number[data][weak_classifier_dimensions[ii]].weight;
														 //更新正規化係數	(全部樣本跑完後會得到完整的正規化係數)
			}
			
		for (int data = 0; data < pos_number + neg_number; data++)	//將所有樣本權重正規化
		{
				*number[data][weak_classifier_dimensions[ii]].weight = *number[data][weak_classifier_dimensions[ii]].weight / z;
		}

		current_best_classifier = 0;
		current_best_classifier_index = 0;
		current_best_classifier_error = 1;
		current_best_classifier_p = 0;

		z = {0};

		fout << weak_classifier_dimensions[ii] << " " << weak_classifier[ii] << " "<< confidence_value[ii] << " " << p[ii] << "\n";		//挑到的弱分類器 及 信心值
		std::cout << weak_classifier_dimensions[ii] << " " << weak_classifier[ii] << " " << confidence_value[ii] << " " << p[ii] << "\n\n";
	}// iteration
	system("pause");
	fout.close();
	pin.close();
	nin.close();
	
}