#include "data_LBP.h"

void data_LBP::copy_lan(data_LBP destiny) {
	weight = destiny.weight;
	pos_neg = destiny.pos_neg;
	data = destiny.data;
}