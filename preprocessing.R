library(mice)

path = "C:/Users/Tanay/College/3rd_year/5th_sem/Machine-Learning/Project/Data"
newdata = read.csv(paste0(path,"/Andhra_dataset2.csv"))

newdata0 = newdata[newdata['reslt']==0,]
newdata1 = newdata[newdata['reslt']==1,]

filled_data0 = mice(newdata0, m = 5, meth = c("","pmm","midastouch","cart","pmm","","midastouch","pmm","cart",""))
filled_data1 = mice(newdata1, m = 5, meth = c("","pmm","midastouch","cart","pmm","","midastouch","pmm","cart",""))

complete_data0 = complete(filled_data0)
complete_data1 = complete(filled_data1)

total <- rbind(complete_data0, complete_data1)


write.csv(total,paste0(path,"/NewImputedDataset.csv"))
