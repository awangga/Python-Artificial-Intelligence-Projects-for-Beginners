import unittest


class TestApp(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_02_rolly_113040087(self):
        from Chapter01.rolly113040087 import preparation,training,testing
        dataset='Chapter01/dataset/student-por.csv'
        d_train_att,d_train_pass,d_test_att,d_test_pass,d_att,d_pass= preparation(dataset)
        t = training(d_train_att,d_train_pass)
        hasiltestingsemua = 	testing(t,d_test_att)
        print('\n hasil testing : ')
        print(hasiltestingsemua)
        ambilsatuhasiltesting = hasiltestingsemua[0]
        self.assertLessEqual(ambilsatuhasiltesting, 1)
        
    def test_02_alvian_1184077(self):
        from Chapter01.alvian1184077 import prepoc, training, predict
        datapath='Chapter01/dataset/covid_19_indonesia_time_series_all.csv'
        data_train_new,data_train_label,data_test_new,data_test_label,d_new,d_label = prepoc(datapath)
        #testing function training
        dt = training(data_train_new, data_train_label)
        hasiltesting = predict(dt, data_test_new)
        print('\n Hasil testing Alvian :')
        print(hasiltesting)
        ambilsatuajatesting = hasiltesting[0]
        self.assertLessEqual(ambilsatuajatesting, 1)