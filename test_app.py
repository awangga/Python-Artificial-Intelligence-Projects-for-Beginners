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
        from Chapter01.alvian1184077 import prepoc, training, testing
        datapath = 'Chapter01/dataset/csv_result-messidor_features.csv'
        d_train_att, d_train_pass, d_test_att, d_test_pass, d_att, d_pass = prepoc(datapath)
        t = training(d_train_att, d_train_pass)
        hasiltestingsemua = testing(t, d_test_att)
        print('\n Hasil testing : ')
        print(hasiltestingsemua)
        ambilsatuhasiltesting = hasiltestingsemua[1]
        self.assertLessEqual(ambilsatuhasiltesting, 2)

    def test_02_mauliddhia_1184101(self):
        from Chapter01.mauliddhia1184101 import prepocesing, training, testing
        datapath = 'Chapter01/dataset/transformed_test.csv'
        d_train_att, d_train_pass, d_test_att, d_test_pass, d_att, d_pass = prepocesing(datapath)
        t = training(d_train_att, d_train_pass)
        hasiltestingsemua = testing(t, d_test_att)
        print('\n Hasil testing : ')
        print(hasiltestingsemua)
        ambilsatuhasiltesting = hasiltestingsemua[1]
        self.assertLessEqual(ambilsatuhasiltesting, 2)

   
