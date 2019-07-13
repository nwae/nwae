#DOC:https://docs.google.com/spreadsheets/d/1rCRpo1-omRYgcHGX8SyEUUxZTGQnKMr7jqvJBRtuGIc/edit#gid=1325600855

library(googlesheets)
library(dplyr)
gs_ls()
my_sheets <- gs_ls()
DF <- gs_title('cx.cn.chatbot.intent-answer-trdata') %>% gs_read(ws='cn.fun88.chatbot.intent-answer-trdata')

####CN####
#your_git_place
setwd('/home/jeffery/文件/git')
DF1 <- read.csv('./mozg.nlp/app.data/intent/traindata/cn.fun88.chatbot.trainingdata.split.csv')
DF2 <- read.csv('./mozg.nlp/app.data/intent/traindata/cn.fun88.chatbot.trainingdata.csv')
#split
df1 <- data.frame(IntentIndex = 14:17,
                  Category = 'cx',
                  Intent = 'APP',
                  Intent.ID = 'cx/APP',
                  Brand = 'fun88',
                  Lang = 'cn',
                  TextLength = 2,
                  Text = c('下载','卸载','版本','解析'),
                  TextSegmented = c('下载','卸载','版本','解析'))
df11 <- data.frame(IntentIndex = 9,
                   Category = 'cx',
                   Intent = 'UI',
                   Intent.ID = 'cx/UI',
                   Brand = 'fun88',
                   Lang = 'cn',
                   TextLength = 2,
                   Text = c('网页'),
                   TextSegmented = c('网页'))

df2 <- rbind(df1,df11,DF1) %>%
  arrange(Category,Intent,Intent.ID) %>%
  rename(`Intent ID` = Intent.ID)
  

write_csv(DF1,'./mozg.nlp/app.data/intent/traindata/cn.fun88.chatbot.trainingdata.split.csv')

write_csv(df2,'./mozg.nlp/app.data/intent/traindata/cn.fun88.chatbot.trainingdata.split.csv')

#training
df1 <- data.frame(Brand = 'fun88',
                  Category = 'cx',
                  Intent = 'APP',
                  Intent.ID = 'cx/APP',
                  Intent.Type = 'internal',
                  Lang = 'cn',
                  Text = c('下载','卸载','版本','解析'))
df11 <- data.frame(Brand = 'fun88',
                   Category = 'cx',
                   Intent = 'UI',
                   Intent.ID = 'cx/UI',
                   Intent.Type = 'internal',
                   Lang = 'cn',
                   Text = c('网页'))

df2 <- rbind(df1,df11,DF2) %>%
  arrange(Category,Intent)

write_csv(DF2,'./mozg.nlp/app.data/intent/traindata/cn.fun88.chatbot.trainingdata.csv')

write_csv(df2,'./mozg.nlp/app.data/intent/traindata/cn.fun88.chatbot.trainingdata.csv')

####TH####
#your_git_place
setwd('/home/jeffery/文件/git')
DF1 <- read.csv('./mozg.nlp/app.data/intent/traindata/th.fun88.chatbot.trainingdata.split.csv')
DF2 <- read.csv('./mozg.nlp/app.data/intent/traindata/th.fun88.chatbot.trainingdata.csv')
#split

df1 <- data.frame(IntentIndex = 1:14,
                  Category = 'cx',
                  Intent = 'APP',
                  Intent.ID = 'cx/APP',
                  Brand = 'fun88',
                  Lang = 'th',
                  TextLength = 2,
                  Text = c('มือถือ','แอพ','เอาออก','ตั้งค่า','IOS','แอนดอย','apple','ระบบ','ซอฟแวร์','แอพพลิเคชั่น','โปรแกรม','โหลด','ดาวน์โหลด','เวอร์ชั่น'),
                  TextSegmented = c('มือถือ','แอพ','เอาออก','ตั้งค่า','IOS','แอนดอย','apple','ระบบ','ซอฟแวร์','แอพพลิเคชั่น','โปรแกรม','โหลด','ดาวน์โหลด','เวอร์ชั่น'))
df1$TextLength <- nchar(as.character(df1$Text))

df11 <- data.frame(IntentIndex = 1:length(strsplit(DF$X10[2],'\n')[[1]]),
                   Category = 'cx',
                   Intent = 'UI',
                   Intent.ID = 'cx/UI',
                   Brand = 'fun88',
                   Lang = 'th',
                   TextLength = 2,
                   Text = strsplit(DF$X10[2],'\n')[[1]],
                   TextSegmented = strsplit(DF$X10[2],'\n')[[1]])
df11$TextLength <- nchar(as.character(df11$Text))

df2 <- rbind(df1,df11,DF1) %>%
  arrange(Category,Intent,Intent.ID) %>%
  rename(`Intent ID` = Intent.ID)
write_csv(df2,'./mozg.nlp/app.data/intent/traindata/th.fun88.chatbot.trainingdata.split.csv')


#training
df1 <- data.frame(Brand = 'fun88',
                  Category = 'cx',
                  Intent = 'APP',
                  Intent.ID = 'cx/APP',
                  Intent.Type = 'internal',
                  Lang = 'th',
                  Text = c('มือถือ','แอพ','เอาออก','ตั้งค่า','IOS','แอนดอย','apple','ระบบ','ซอฟแวร์','แอพพลิเคชั่น','โปรแกรม','โหลด','ดาวน์โหลด','เวอร์ชั่น'))
df11 <- data.frame(Brand = 'fun88',
                   Category = 'cx',
                   Intent = 'UI',
                   Intent.ID = 'cx/UI',
                   Intent.Type = 'internal',
                   Lang = 'th',
                   Text = strsplit(DF$X10[2],'\n')[[1]])

df2 <- rbind(df1,df11,DF2) %>%
  arrange(Category,Intent) 
write_csv(df2,'./mozg.nlp/app.data/intent/traindata/th.fun88.chatbot.trainingdata.csv')
