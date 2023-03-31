library(readxl)
library(tidyverse)
library(ggplot2)

library(reshape2)

# Importing ---------------------------------------------------------------

df_contratos<-readxl::read_excel("../BancoFamilia/BF_Dados_Contratos.xlsx")
df_Socio<-readxl::read_excel("../BancoFamilia/BF_Dados_Socio_Economicos.xlsx")




# Deixando apenas id acima de 40 mil --------------------------------------


x<-df_contratos$CONTRATO>40000

df_contratos<-df_contratos[x,]
rm(x)


str(df_Socio)


#Join das tabelas para juntar os contratos com os dados Socio demograficos
# A situação para ser considerada perda tem que estar em Aberto e ter mais de 180 dias de atraso no pgto

# Mudando para lowe rcase as colunas
names(df_contratos) <- tolower(names(df_contratos))

bigdf <- left_join(df_contratos,df_Socio, by = "identificação" )

bigdf<-drop_na(bigdf)


# Mantendo linhas com a combinação de id e data de contrato mais recentes
bigdf<-bigdf %>% 
  group_by(identificação) %>%
  slice(which.max(as.Date(data_contrato, '%m/%d/%Y')))

names(bigdf)

# Análise das variáveis ---------------------------------------------------



# Tratamento de dados -----------------------------------------------------

bigdf$renda_cliente <- as.numeric(bigdf$renda_cliente)
bigdf$valor_emprestimo <- as.numeric(bigdf$valor_emprestimo)
bigdf$prestacao<-as.numeric(bigdf$prestacao)
bigdf$atraso_maximo<-as.numeric(bigdf$atraso_maximo)
df1<-bigdf %>% select(-c(bairro,cep,identificação,
                         contrato,data_inclusao,atividade,data_vencimento_final,
                         data_vencimento_incial, melhor_data_vencimento,data_liberacao,data_contrato,data_liquidacao))

# Tentativa 2 -------------------------------------------------------------


# Apenas num[ericas]
nums <- unlist(lapply(df1, is.numeric), use.names = FALSE)  
x<-names(df1[ , nums])


numes<-df1[ , nums]
meltData <- melt(numes)
boxplot(data=meltData, value~variable)


p <- ggplot(meltData, aes(factor(variable), value)) 
p + geom_boxplot() + facet_wrap(~variable, scale="free")



GGally::ggpairs(df1)


# Definir as variáveis necessárias pra clusterização ----------------------





 



