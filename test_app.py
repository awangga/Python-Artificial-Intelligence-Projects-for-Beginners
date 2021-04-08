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
        hasiltestingsemua =     testing(t,d_test_att)
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
        hasiltestingsemua =     testing(it,anak_test_att)
        print('\n hasil testing : ')
        print(hasiltestingsemua)
        ambilsatuhasiltesting = hasiltestingsemua[1]
        self.assertLessEqual(ambilsatuhasiltesting, 8)

    def test_02_yuki_1184053(self):
        from Chapter01.yuki1184053 import prepoc,training,testing
        dataset='Chapter01/dataset/bankruptcy_prediction.csv'
        pencak_train_att,pencak_train_pass,pencak_test_att,pencak_test_pass,pencak_att,pencak_pass= prepoc(dataset)
        silat = training(pencak_train_att,pencak_train_pass)
        hasiltestingsemua =     testing(silat,pencak_test_att)
        print('\n hasil testing : ')
        print(hasiltestingsemua)
        ambilsatuhasiltesting = hasiltestingsemua[0]
        self.assertLessEqual(ambilsatuhasiltesting, 1)

    def test_02_nurul_1184038(self):
        from Chapter01.nurul1184038 import prepoc, training, testing
        datapath = 'Chapter01/dataset/student-mat-pass-or-fail.csv'
        d_train_att, d_train_pass, d_test_att, d_test_pass, d_att, d_pass = prepoc(datapath)
        t = training(d_train_att, d_train_pass)
        hasiltestingsemua = testing(t, d_test_att)
        print('\n Hasil testing : ')
        print(hasiltestingsemua)
        ambilsatuhasiltesting = hasiltestingsemua[0]
        self.assertLessEqual(ambilsatuhasiltesting, 1)
        
    def test_03_aditya_1184090(self):
        from Chapter02.Aditya1184090 import preparation,training,testing
        #data
        data = preparation()
        #train data
        train = data.pop(0)
        d_train_attribute = train.pop(0)
        d_train_var = train.pop(0)
        #test data
        test = data.pop(0)
        d_test_attribute = test.pop(0)
        d_test_var = test.pop(0)
        #training
        t = training(d_train_attribute, d_train_var)
        #predict
        result = testing(t,d_test_attribute)
        print("result : ")
        print(result)
        print('Score:', t.score(d_test_attribute, d_test_var))
        self.assertLessEqual(result[0], 2)
        
    def test_03_yusuf_1184026(self):
        from Chapter02.Yusuf1184026 import preparation, training, testing
        dataset = 'Chapter01/dataset/car.txt'
        # testing function preparation
        df_train_att, df_train_label, df_test_att, df_test_label, df_att, df_label = preparation(dataset)
        # testing function training
        clf = training(df_train_att, df_train_label)
        # testing function testing
        hasil = testing(clf, df_test_att.head())
        # hasil testing
        print('\nhasil testing Yusuf :', hasil)
        print('Score:', clf.score(df_test_att, df_test_label))
    
    def test_03_rizalramadhan_1184033(self):
        from Chapter02.rizalramadhan1184033 import prepoc, training, testing
        datapath = 'Chapter01/dataset/mobile.txt'
        nanas_train_att, nanas_train_pass, nanas_test_att, nanas_test_pass, nanas_att, nanas_pass = prepoc(datapath)
        apel = training(nanas_train_att, nanas_train_pass)
        hasiltestingsemua = testing(apel, nanas_test_att)
        print('\n Hasil testing : ')
        print(hasiltestingsemua)
        print('Score:', apel.score(nanas_test_att, nanas_test_pass))
        ambilsatuhasiltesting = hasiltestingsemua[1]
        self.assertLessEqual(ambilsatuhasiltesting, 3)
        
    def test_03_kevin_1184049(self):
        from Chapter02.kevin1184049 import preparation, training, testing
        datapath = 'Chapter01/dataset/phishing.txt'
        d_train_att, d_train_pass, d_test_att, d_test_pass, d_att, d_pass = preparation(datapath)
        t = training(d_train_att, d_train_pass)
        hasiltestingsemua = testing(t, d_test_att)
        print('\n Hasil testing : ')
        print(hasiltestingsemua)
        print('Score:', t.score(d_test_att, d_test_pass))
        ambilsatuhasiltesting = hasiltestingsemua[1]
        self.assertLessEqual(ambilsatuhasiltesting, 2)
        
    def test_03_dian_1184095(self):
        from Chapter02.dian1184095 import prepoc, training, testing
        datapath = 'Chapter01/dataset/new_test.txt'
        d_train_att, d_train_pass, d_test_att, d_test_pass, d_att, d_pass = prepoc(datapath)
        t = training(d_train_att, d_train_pass)
        hasiltestingsemua = testing(t, d_test_att)
        print('\n Hasil testing : ')
        print(hasiltestingsemua)
        ambilsatuhasiltesting = hasiltestingsemua[1]
        self.assertLessEqual(ambilsatuhasiltesting, 2)
        print('Score:', t.score(d_test_att, d_test_pass))
        
    def test_03_YukiArdiansyah_1184053(self):
        from Chapter02.YukiArdiansyah1184053 import preparation,training,testing
        #data
        data = preparation()
        #train data
        train = data.pop(0)
        yuki_train_attribute = train.pop(0)
        yuki_train_var = train.pop(0)
        #test data
        test = data.pop(0)
        yuki_test_attribute = test.pop(0)
        yuki_test_var = test.pop(0)
        #training
        ardi = training(yuki_train_attribute, yuki_train_var)
        #predict
        result = testing(ardi,yuki_test_attribute)
        print("Hasil : ")
        print(result)
        print('Score:', ardi.score(yuki_test_attribute, yuki_test_var))
        self.assertLessEqual(result[0], 2)
        
    def test_03_almi_1184043(self):
        from Chapter02.almi1184043 import preparation,training,testing
        data = preparation()
        train = data.pop(0)
        d_train_attribute = train.pop(0)
        d_train_var = train.pop(0)
        test = data.pop(0)
        d_test_attribute = test.pop(0)
        d_test_var = test.pop(0)
        t = training(d_train_attribute, d_train_var)
        result = testing(t,d_test_attribute.head())
        print("result : ")
        print(result)
        print('Score:', t.score(d_test_attribute, d_test_var))
        self.assertLessEqual(result[1], 7)
        
    def test_03_helmiazhar_1184013(self):
        from Chapter02.helmiazhar1184013 import preparation,training,testing
        #data
        data = preparation()
        #train data
        train = data.pop(0)
        d_train_attribute = train.pop(0)
        d_train_var = train.pop(0)
        #test data
        test = data.pop(0)
        d_test_attribute = test.pop(0)
        d_test_var = test.pop(0)
        #training
        t = training(d_train_attribute, d_train_var)
        #predict
        result = testing(t,d_test_attribute.head())
        print("result : ")
        print(result)
        print('Score:', t.score(d_test_attribute, d_test_var))
        self.assertLessEqual(result[0], 1)
        
    def test_03_nurulkamila_1184038(self):
        from Chapter02.nurul1184038 import preparation,training,testing
        #data
        data = preparation()
        #train data
        train = data.pop(0)
        d_train_attribute = train.pop(0)
        d_train_var = train.pop(0)
        #test data
        test = data.pop(0)
        d_test_attribute = test.pop(0)
        d_test_var = test.pop(0)
        #training
        t = training(d_train_attribute, d_train_var)
        #predict
        result = testing(t,d_test_attribute.head())
        print("result : ")
        print(result)
        print('Score:', t.score(d_test_attribute, d_test_var))
        self.assertLessEqual(result[0], 1)
        
    def test_03_murnialestari_1184006(self):
        from Chapter02.murnialestari1184006 import preparation, training, testing

        datasetpath = 'Chapter01/dataset/bill_authentication.txt'
        # testing function preparation
        df_train_att, df_train_label, df_test_att, df_test_label, df_att, df_label = preparation(datasetpath)
        # testing function training
        clf = training(df_train_att, df_train_label)
        # testing function testing
        hasiltesting = testing(clf, df_test_att.head())
        # hasil
        print('\nhasil testing murnia : ')
        print(hasiltesting)
        ambilsatuhasiltesting = hasiltesting[0]
        self.assertLessEqual(ambilsatuhasiltesting, 1)                                    
        print('Score:', clf.score(df_test_att, df_test_label))
        
    def test_03_dindaanik_1184003(self):
        from Chapter02.dindaanik1184003 import preparation, training, testing

        datasetpath = 'Chapter01/dataset/College_admission.txt'
        # testing function preparation
        df_train_att, df_train_label, df_test_att, df_test_label, df_att, df_label = preparation(datasetpath)
        # testing function training
        clf = training(df_train_att, df_train_label)
        # testing function testing
        hasiltesting = testing(clf, df_test_att.head())
        # hasil
        print('\nhasil testing dinda : ')
        print(hasiltesting)
        ambilsatuhasiltesting = hasiltesting[0]
        self.assertLessEqual(ambilsatuhasiltesting, 1)
        print('Score:', clf.score(df_test_att, df_test_label))
        
    def test_03_annisa_1184071(self):
        from Chapter02.annisa1184071 import preparation, training, testing

        datasetpath = 'Chapter01/dataset/future_50.txt'
        # testing function preparation
        df_train_att, df_train_label, df_test_att, df_test_label, df_att, df_label = preparation(datasetpath)
        # testing function training
        clf = training(df_train_att, df_train_label)
        # testing function testing
        hasiltesting = testing(clf, df_test_att.head())
        # hasil
        print('\nhasil testing annisa : ')
        print(hasiltesting)
        ambilsatuhasiltesting = hasiltesting[0]
        self.assertLessEqual(ambilsatuhasiltesting, 1)
        print('Score:', clf.score(df_test_att, df_test_label))

    def test_03_nuha_1184085(self):
        from Chapter02.nuha1184085 import preparation, training, testing
        datasetpath = 'Chapter01/dataset/e-shop_clothing.txt'       
        # testing function preparation
        df_train_att, df_train_label, df_test_att, df_test_label, df_att, df_label = preparation(datasetpath)
        # testing function training
        clf = training(df_train_att, df_train_label)
        # testing function testing
        hasiltesting = testing(clf, df_test_att.head())
        # hasil
        print('\nhasil testing nuha : ')
        print(hasiltesting)
        print('Score:', clf.score(df_test_att, df_test_label))
        self.assertLessEqual(hasiltesting[1], 4) 
        
    def test_03_gany_1184008(self):
        from Chapter02.Gany1184008 import preparation, training, testing
        datasetpath = 'Chapter01/dataset/AirQualiityUCI.txt'
        # testing function preparation
        tes_train_att,tes_train_pass,tes_test_att,tes_test_pass,tes_att,tes_pass= preparation(datasetpath)
        # testing function training
        x = training(tes_train_att, tes_train_pass)
        # testing function testing
        hasil = testing(x, tes_test_att.head())
        # hasil testing
        print('\nhasil testing gany :', hasil)
        print('Score:', x.score(tes_test_att, tes_test_pass)) 
        self.assertLessEqual(hasil[1], 2)
        
    def test_03_dyah_1184098(self):
        from Chapter02.dyah1184098 import prepoc, training, testing
        datapath = 'Chapter01/dataset/ModeChoice (Data to Study Travel Mode Choice).txt'
        d_train_att, d_train_pass, d_test_att, d_test_pass, d_att, d_pass = prepoc(datapath)
        t = training(d_train_att, d_train_pass)
        hasiltestingsemua = testing(t, d_test_att.head())
        print('\n Hasil testing : ')
        print(hasiltestingsemua)
        ambilsatuhasiltesting = hasiltestingsemua[1]
        self.assertLessEqual(ambilsatuhasiltesting, 6)
        print('Score:', t.score(d_test_att, d_test_pass))
        
    def test_03_Alvian_1184077(self):
        from Chapter02.Alvian1184077 import preparation, training, testing
        datasetpath = 'Chapter01/dataset/house-votes-84.txt'
        f_train_att, f_train_label, f_test_att, f_test_label, f_att, f_label = preparation(datasetpath)
        # testing dari fungsi traning
        clf = training(f_train_att, f_train_label)
        # testing dari fungsi testing
        hasiltesting = testing(clf, f_test_att.head())

        # hasil testing yang dilakukan
        print(' testing : ')
        print(hasiltesting)
        ambilsatuhasiltesting = hasiltesting[0]
        self.assertLessEqual(ambilsatuhasiltesting, 1)
        print('Hasil ', clf.score(f_test_att, f_test_label))

    def test_03_Mauliddhia_1184101(self):
        from Chapter02.Mauliddhia1184101 import preparation, training, testing

        datasetpath = 'Chapter01/dataset/kr-vs-kp.txt'
        f_train_att, f_train_label, f_test_att, f_test_label, f_att, f_label = preparation(datasetpath)
        # testing dari fungsi traning
        clf = training(f_train_att, f_train_label)
        # testing dari fungsi testing
        hasiltesting = testing(clf, f_test_att.head())

        # hasil testing yang dilakukan
        print(' testing : ')
        print(hasiltesting)
        ambilsatuhasiltesting = hasiltesting[0]
        self.assertLessEqual(ambilsatuhasiltesting, 1)
        print('Hasil ', clf.score(f_test_att, f_test_label))
        
        
    def test_04_aditya_1184090(self):
        from Chapter03.Aditya1184090 import preparation,training,testing
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
        print('Score:', t.score(dfs_test_att, dfs_test_win))
        self.assertLessEqual(result[0], 1)
        
    def test_04_yusuf_1184026(self):
        from Chapter03.Yusuf1184026 import preparation, training, testing
        data = preparation()

        train = data.pop(0)
        test = data.pop(0)

        trainAttr = train.pop(0)
        trainVar = train.pop(0)

        testAttr = test.pop(0)
        testVar = test.pop(0)

        t = training(trainAttr, trainVar)

        result = testing(t, testAttr)
        print('result : ')
        print(result)
        print("score : ",t.score(testAttr, testVar))
        self.assertLessEqual(result[0], 1)

    def test_04_alvian_1184077(self):
        from Chapter03.alvian1184077 import preparation, training, testing
        data = preparation()

        train = data.pop(0)
        test = data.pop(0)

        trainAttr = train.pop(0)
        trainVar = train.pop(0)

        testAttr = test.pop(0)
        testVar = test.pop(0)

        t = training(trainAttr, trainVar)

        result = testing(t, testAttr)
        print('result : ')
        print(result)
        print("score : ", t.score(testAttr, testVar))
        self.assertLessEqual(result[0], 1)

    def test_04_mauliddhia_1184101(self):
        from Chapter03.Mauliddhia1184101 import preparation, training, testing
        data = preparation()

        train = data.pop(0)
        test = data.pop(0)

        trainAttr = train.pop(0)
        trainVar = train.pop(0)

        testAttr = test.pop(0)
        testVar = test.pop(0)

        t = training(trainAttr, trainVar)

        result = testing(t, testAttr)
        print('result : ')
        print(result)
        print("score : ", t.score(testAttr, testVar))
        self.assertLessEqual(result[0], 1)
        
    def test_04_nurul_1184038(self):
        from Chapter03.nurul1184038 import preparation,training,testing
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
        print('Score:', t.score(dfs_test_att, dfs_test_win))
        self.assertLessEqual(result[0], 1)
        
    def test_04_kevin_1184049(self):
        from Chapter03.kevin1184049 import prepoc, training, testing
        datapath = 'Chapter01/dataset/spam_ham_dataset.csv'
        d_train_att, d_train_pass, d_test_att, d_test_pass= prepoc(datapath)
        t = training(d_train_att, d_train_pass)
        hasiltestingsemua = testing(t, d_test_att)
        print('\n Hasil testing : ')
        print(hasiltestingsemua)
        print('Score:', t.score(d_test_att, d_test_pass))
        ambilsatuhasiltesting = hasiltestingsemua[1]
        self.assertLessEqual(ambilsatuhasiltesting, 1)
        
    def test_04_helmiazhar_1184013(self):
        from Chapter03.helmiazhar1184013 import preparation,training,testing
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
        print('Score:', t.score(dfs_test_att, dfs_test_win))
        self.assertLessEqual(result[1], 5)

    def test_04_rizalramadhan_1184033(self):
        from Chapter03.rizalramadhan1184033 import prepoc, training, testing
        datapath = 'Chapter01/dataset/emailspam.csv'
        nanas_train_att, nanas_train_pass, nanas_test_att, nanas_test_pass= prepoc(datapath)
        apel = training(nanas_train_att, nanas_train_pass)
        hasiltestingsemua = testing(apel, nanas_test_att)
        print('\n Hasil testing : ')
        print(hasiltestingsemua)
        print('Score:', apel.score(nanas_test_att, nanas_test_pass))
        ambilsatuhasiltesting = hasiltestingsemua[1]
        self.assertLessEqual(ambilsatuhasiltesting, 1)
        
    def test_04_murnia_1184006(self):
        from Chapter03.murnialestari1184006 import prepoc, training, testing
        datapath = 'Chapter01/dataset/Costumes_Amazon.csv'
        d_train_att, d_train_pass, d_test_att, d_test_pass= prepoc(datapath)
        t = training(d_train_att, d_train_pass)
        hasiltestingsemua = testing(t, d_test_att)
        print('\n Hasil testing : ')
        print(hasiltestingsemua)
        print('Score:', t.score(d_test_att, d_test_pass))
        ambilsatuhasiltesting = hasiltestingsemua[1]
        self.assertLessEqual(ambilsatuhasiltesting, 5)
        
    def test_04_yukiardiansyah_1184053(self):
        from Chapter03.YukiArdiansyah1184053 import preparation,training,testing
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
        print('Score:', t.score(dfs_test_att, dfs_test_win))
        self.assertLessEqual(result[1], 5)
        
    def test_04_dimas_1184081(self):
        from Chapter03.dimas1184081 import preparation,training,testing
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
        print('Score:', t.score(dfs_test_att, dfs_test_win))
        self.assertLessEqual(result[1], 5)
        
    def test_04_annisa_1184071(self):
        from Chapter03.annisa1184071 import preparation,training,testing
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
        print('Score:', t.score(dfs_test_att, dfs_test_win))
        self.assertLessEqual(result[0], 1)
        
    def test_04_dianm_1184095(self):
        from Chapter03.dianm1184095 import prepoc, training, testing
        datapath = 'Chapter01/dataset/lapak.csv'
        d_train_att, d_train_pass, d_test_att, d_test_pass= prepoc(datapath)
        t = training(d_train_att, d_train_pass)
        hasiltestingsemua = testing(t, d_test_att)
        print('\n Hasil testing : ')
        print(hasiltestingsemua)
        print('Score:', t.score(d_test_att, d_test_pass))
        ambilsatuhasiltesting = hasiltestingsemua[0]
        self.assertLessEqual(ambilsatuhasiltesting, 1)
        
    def test_04_dinda_1184003(self):
        from Chapter03.dinda1184003 import prepoc, training, testing
        datapath = 'Chapter01/dataset/hotel_reviews.csv'
        d_train_att, d_train_pass, d_test_att, d_test_pass= prepoc(datapath)
        t = training(d_train_att, d_train_pass)
        hasiltestingsemua = testing(t, d_test_att)
        print('\n Hasil testing : ')
        print(hasiltestingsemua)
        print('Score:', t.score(d_test_att, d_test_pass))
        ambilsatuhasiltesting = hasiltestingsemua[1]
        self.assertLessEqual(ambilsatuhasiltesting, 5)
        
    def test_04_nuha_1184085(self):
        from Chapter03.nuha1184085 import preparation, training, testing
        datasetpath = 'Chapter01/dataset/apple-twitter-sentiment-texts.csv'
        d_train_att, d_train_pass, d_test_att, d_test_pass= preparation(datasetpath)
        t = training(d_train_att, d_train_pass)
        hasiltestingsemua = testing(t, d_test_att)
        print('\n Hasil testing nuha : ')
        print(hasiltestingsemua)
        print('Score:', t.score(d_test_att, d_test_pass))
        ambilsatuhasiltesting = hasiltestingsemua[1]
        self.assertLessEqual(ambilsatuhasiltesting, 1)
        
    def test_04_Nella_1184017(self):
        from Chapter03.Nella_1184017 import prepoc, training, testing
        datapath = 'Chapter01/dataset/Review.csv'
        d_train_att, d_train_pass, d_test_att, d_test_pass= prepoc(datapath)
        t = training(d_train_att, d_train_pass)
        hasiltestingsemua = testing(t, d_test_att)
        print('\n Hasil testing nella : ')
        print(hasiltestingsemua)
        print('Score:', t.score(d_test_att, d_test_pass))
        ambilsatuhasiltesting = hasiltestingsemua[1]
        self.assertLessEqual(ambilsatuhasiltesting, 5)
        
    def test_04_gany_1184008(self):
        from Chapter03.gany1184008 import preparation, training, testing
        datasetpath = 'Chapter01/dataset/Indonesian Sentiment Twitter Dataset Labeled.csv'
        d_train_att, d_train_pass, d_test_att, d_test_pass= preparation(datasetpath)
        t = training(d_train_att, d_train_pass)
        hasiltestingsemua = testing(t, d_test_att)
        print('\n Hasil testing gany : ')
        print(hasiltestingsemua)
        print('Score:', t.score(d_test_att, d_test_pass))
        ambilsatuhasiltesting = hasiltestingsemua[1]
        self.assertLessEqual(ambilsatuhasiltesting, 1)
        
    def test_04_dyah_1184098(self):
        from Chapter03.dyah1184098 import prepoc, training, testing
        datapath = 'Chapter01/dataset/reviews_pilot.csv'
        d_train_att, d_train_pass, d_test_att, d_test_pass= prepoc(datapath)
        t = training(d_train_att, d_train_pass)
        hasiltestingsemua = testing(t, d_test_att)
        print('\n Hasil testing : ')
        print(hasiltestingsemua)
        print('Score:', t.score(d_test_att, d_test_pass))
        ambilsatuhasiltesting = hasiltestingsemua[1]
        self.assertLessEqual(ambilsatuhasiltesting, 5)
        
    def test_04_almi_1184043(self):
        from Chapter03.almi1184043 import preparation,training,testing
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
        print('Score:', t.score(dfs_test_att, dfs_test_win))
        self.assertLessEqual(result[1], 5)

    def test_04_muhamri_1184100(self):
        from Chapter03.MuhAmri_1184100 import preparation,training,testing
        data = preparation()
        
        train = data.pop(0)
        buy_train_attribute = train.pop(0)
        buy_train_win = train.pop(0)
        
        test = data.pop(0)
        buy_test_attribute = test.pop(0)
        buy_test_win = test.pop(0)
        
        w = training(buy_train_attribute,buy_train_win)
        
        result = testing(w,buy_test_attribute)
        print('result : ')
        print(result)
        print("score : ", w.score(buy_test_attribute,buy_test_win))
        self.assertLessEqual(result[0], 1)
        
    def test_04_ilhamdwi_1184057(self):
        from Chapter03.ilhamdwi1184057 import preparation,training,testing
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
        print('Score:', t.score(dfs_test_att, dfs_test_win))
        self.assertLessEqual(result[1], 5)