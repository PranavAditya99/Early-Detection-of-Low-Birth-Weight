library(mice)

path = "path/to/data"
newdata = read.csv(paste0(path,"/Andhra_dataset2.csv"))


filled_data = mice(newdata, m = 5, meth = c("","pmm","midastouch","cart","pmm","","midastouch","pmm","cart",""))
complete_data = complete(filled_data)
write.csv(complete_data,paste0(path,"/ImputedAndhraData.csv"))
