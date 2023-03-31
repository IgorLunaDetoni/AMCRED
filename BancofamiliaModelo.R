library(readxl)
library(tidyverse)
library(ggplot2)
library(klaR)
library(cluster)
library(Matrix)
library(recipes)
library(writexl)
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


# Definir as variáveis necessárias pra clusterização ----------------------

gg <- df1 %>% select(c(renda_cliente,situacao,valor_emprestimo,renegociado,valor_solicitado,prazo_em_meses,
                       melhor_valor_parcela, tipo_atividade,tempo_atividade,total_receitas,numero_de_pessoas_na_casa,
                       situacao_do_imovel,tempo_de_residancia__anos,media_dos_faturamentos))

GGally::ggpairs(gg)

# Segunda limpeza ---------------------------------------------------------


# transformando em dummies ------------------------------------------------

dummies <- gg %>%  recipe(situacao~.) %>%
  step_dummy(c(situacao,tipo_atividade,tempo_atividade,situacao_do_imovel,tempo_de_residancia__anos), one_hot = TRUE) %>% 
  step_normalize(c(renda_cliente,valor_emprestimo,valor_solicitado,melhor_valor_parcela,total_receitas)) %>% 
  prep() %>% bake(gg)



# PCA ---------------------------------------------------------------------

dummies<-na.omit(dummies)

pCA_ <-prcomp(dummies)
plot(pCA_$sdev^2/sum(pCA_$sdev^2),xlab = "PCA", ylab = "Proporção da variância", type ="l")

plot(pCA_$sdev, xlab = "PC", ylab = "Eigenvalues", type = "l")
 


# Kmodes ------------------------------------------------------------------

result1<-kmodes(dummies, 6, iter.max = 10, weighted = FALSE)
result1$size
result1$modes

#Colorindo de acordo com os clusters
plot(pCA_$x[,1],pCA_$x[,6],col=result1$cluster)



plot(pCA_$sdev^2/sum(pCA_$sdev^2),xlab = "PCA", ylab = "Proporção da variância", type ="l")





