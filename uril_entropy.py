''' Unless stated otherwise, all software is provided free of charge. 
As well, all software is provided on an "as is" basis without warranty 
of any kind, express or implied. Under no circumstances and under no legal 
theory, whether in tort, contract, or otherwise, shall Liang Zhang be liable 
to you or to any other person for any indirect, special, incidental, 
or consequential damages of any character including, without limitation, 
damages for loss of goodwill, work stoppage, computer failure or malfunction, 
or for any and all other damages or losses. If you do not agree with these terms, 
then you are advised to not use the software.'''

import pandas as pd
import numpy as np


class EntropyCls():
    def __init__(self):
        pass

    def _entropy_single_variable(self, X):
        X = list(X)
        probs = [list(X).count(c) * 1.0 / len(X) for c in set(X)]
        return np.sum(-p * np.log2(p) for p in probs)

    def _joint_entropy_two_variables(self, X, Y):
        """
        :param X:
        :param Y:
        :return: H(X|Y)
        """
        data = pd.DataFrame({'X': list(X), 'Y': list(Y)})
        joint_entropy = 0
        for y_item in set(data['Y']):
            temp_data = data[data['Y'] == y_item]
            temp_list = temp_data['X']
            entropy = self._entropy_single_variable(temp_list)
            p_y = len(temp_data) / len(data)
            joint_entropy += p_y * entropy
        return joint_entropy

    def _get_information_grain(self, X, Y):
        """
        :param X:
        :param Y:
        :return: IG(X|Y) = H(X) - H(X|Y)
        """
        return self._entropy_single_variable(X) - self._joint_entropy_two_variables(X, Y)

    def _get_sym_uncertity(self, X, Y):
        """
        :param X:
        :param Y:
        :return: SU(X,Y) = 2[IG(X|Y)/(H(X)+H(Y))]
        """
        return 2 * (
            self._get_information_grain(X, Y) / (self._entropy_single_variable(X) + self._entropy_single_variable(Y)))

    def get_sym_uncertity_matrix(self, data):
        """
        :param data: pandas data
        :return:
        """
        name_list = list(data)
        result = pd.DataFrame(index=name_list, columns=name_list)
        for c_idx, c_item in enumerate(name_list):
            for r_idx, r_item in enumerate(name_list):
                if c_idx > r_idx:
                    result.loc[r_item, c_item] = self._get_sym_uncertity(data[r_item], data[c_item])

        return result

    def get_coeff(self, data):
        name_list = list(data)
        result = pd.DataFrame(index=name_list, columns=name_list)
        for c_idx, c_item in enumerate(name_list):
            for r_idx, r_item in enumerate(name_list):
                if c_idx > r_idx:
                    result.loc[r_item, c_item] = abs(np.corrcoef(data[r_item], data[c_item])[0][1])
        return result

def print_assistment2009():
    data = pd.read_csv(
        "./data/assistment2009/attempt_level correct attempt_level time_level correct first_action correct first_action time_level correct skill_id correct time_level correct_large_.csv")

    temp_data = data[
        ['correct', 'first_action', 'time_level', 'attempt_level', 'attempt_level correct', 'first_action correct',
         'time_level correct','attempt_level time_level correct','first_action time_level correct']]
    r1 = EntropyCls().get_sym_uncertity_matrix(temp_data)
    r2 = EntropyCls().get_coeff(temp_data)

    result = pd.concat([r1, r2])
    result.to_csv('./result/assistment2009/correlationship_add_3.csv')
    print(result)

def print_cmu():
    data = pd.read_csv(
        "./data/cmu_stat_f2011/skill_id correct skill_id time_level time_level correct_large_.csv")

    temp_data = data[
        ['correct', 'first_action', 'time_level', 'attempt_level', "time_level correct","skill_id time_level","skill_id correct"]]
    r1 = EntropyCls().get_sym_uncertity_matrix(temp_data)
    r2 = EntropyCls().get_coeff(temp_data)

    result = pd.concat([r1, r2])
    result.to_csv('./result/cmu_stat_f2011/correlationship_add_3.csv')
    print(result)

if __name__ == "__main__":
    #print_assistment2009()
    print_cmu()

