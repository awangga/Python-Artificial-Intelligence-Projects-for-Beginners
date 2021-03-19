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
        
    def test_02_aditya_1184090(self):
        from Chapter01.Aditya1184090 import preparation,training,testing
        #data
        data = preparation()
        #train data
        train = data.pop(0)
        dfs_train_att = train.pop(0)
        dfs_train_win = train.pop(0)
        #test data
        test = data.pop(0)
        dfs_test_att = test.pop(0)
        dfs_test_win = test.pop(0)
        #training
        t = training(dfs_train_att, dfs_train_win)
        #predict
        result = testing(t,dfs_test_att)
        print("result : ")
        print(result)
        self.assertLessEqual(result[0], 2)

    def test_02_rizal_1184033(self):
        from Chapter01.rizalramadhan1184033 import prepoc, training, testing
        datapath = 'Chapter01/dataset/train.csv'
        nanas_train_att, nanas_train_pass, nanas_test_att, nanas_test_pass, nanas_att, nanas_pass = prepoc(datapath)
        apel = training(nanas_train_att, nanas_train_pass) 
        hasiltestingsemua = testing(apel, nanas_test_att) 
        print('\n Hasil testing : ') 
        print(hasiltestingsemua) 
        ambilsatuhasiltesting = hasiltestingsemua[1] 
        self.assertLessEqual(ambilsatuhasiltesting, 1)
        
    def test_02_dimas_1184081(self):
        from Chapter01.dimas1184081 import prepoc,training,testing
        dataset='Chapter01/dataset/Pokedex_Condensed_Numeric_Dataset.csv'
        anak_train_att,anak_train_pass,anak_test_att,anak_test_pass,anak_att,anak_pass= prepoc(dataset)
        it = training(anak_train_att,anak_train_pass)
        hasiltestingsemua = 	testing(it,anak_test_att)
        print('\n hasil testing : ')
        print(hasiltestingsemua)
        ambilsatuhasiltesting = hasiltestingsemua[1]
        self.assertLessEqual(ambilsatuhasiltesting, 8)

    def test_02_dicky_1184019(self):
        from Chapter01.dicky_1184019 import prepoc,training,testing
        dataset='Chapter01/dataset/submission_rf.csv'
        d_train_att,d_train_pass,d_test_att,d_test_pass,d_att,d_pass= prepoc(dataset)
        t = training(d_train_att,d_train_pass)
        hasiltestingsemua = 	testing(t,d_test_att)
        print('\n hasil testing : ')
        print(hasiltestingsemua)
        ambilsatuhasiltesting = hasiltestingsemua[0]
        self.assertLessEqual(ambilsatuhasiltesting, 4) 
        
a= TestApp()
a.test_02_dicky_1184019()
