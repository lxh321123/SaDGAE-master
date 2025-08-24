@echo off
setlocal EnableDelayedExpansion

@REM echo. > log.txt
set /a "num[1] = 9"
set /a "num[2] = 11"
set /a "num[3] = 4"
set /a "num[4] = 5"
set /a "num[5] = 7"
set /a "num[6] = 4"
set /a "num[7] = 5"
set /a "num[8] = 6"
set /a "num[9] = 4"
set /a "num[10] = 2"
set /a "num[11] = 9"
set /a "num[12] = 8"

set "name1=Muraro"
set "name2=Pollen"
set "name3=Quake_10x_Bladder"
set "name4=Quake_Smart-seq2_Diaphragm"
set "name5=Romanov"
set "name6=Klein"
set "name7=goolam"
set "name8=Yan"
set "name9=Xin"
set "name10=Wang_Lung"
set "name11=Zeisel"
set "name12=10X_PBMC"

@REM 消融实验的neighbors和hvg
set /a "hv[1]=1500"
set /a "hv[2]=1000"
set /a "hv[3]=1500"
set /a "hv[4]=1000"
set /a "hv[5]=500"
set /a "hv[6]=1500"
set /a "hv[7]=1500"
set /a "hv[8]=1500"
set /a "hv[9]=1500"
set /a "hv[10]=1000"
set /a "hv[11]=2000"
set /a "hv[12]=500"


set /a "nei[1]=9"
set /a "nei[2]=12"
set /a "nei[3]=13"
set /a "nei[4]=6"
set /a "nei[5]=5"
set /a "nei[6]=15"
set /a "nei[7]=13"
set /a "nei[8]=5"
set /a "nei[9]=13"
set /a "nei[10]=8"
set /a "nei[11]=12"
set /a "nei[12]=14"



for /L %%i in (1,1,9) do (
    echo dataset=!name%%i!, hvg=!hv[%%i]!, n_clusters=!num[%%i]!, neighbors=!nei[%%i]!
    echo pretrain AutoEncoder:
    python ./ae/main.py --name=!name%%i! --hvg=!hv[%%i]! --n_clusters=!num[%%i]!
    echo -------------------------------------------------------------------------------------------

    echo pretrain GNN:
    python ./GNN/main.py --name=!name%%i! --hvg=!hv[%%i]! --neighbors=!nei[%%i]! --n_clusters=!num[%%i]!

    echo pretrain:
    echo pretrain: >> log.txt
    python ./pretrain/main.py --name=!name%%i! --n_clusters=!num[%%i]! --hvg=!hv[%%i]! --neighbors=!nei[%%i]! >> log.txt
    echo ------------------------------------------------------------------------------------

    echo formal train:
    echo datasset=!name%%i!, hvg=!hv[%%i]!, neighbors=!nei[%%i]! >> log.txt
    echo formal train: >> log.txt
    python ./scDeGAEsa/main.py --name=!name%%i! --n_clusters=!num[%%i]! --hvg=!hv[%%i]! --neighbors=!nei[%%i]! >> log.txt
    echo completed! result: %errorlevel%
    echo -------------------------------------------------------------------------------------
    echo ------------------------------------------------------------------------------------- >> log.txt

)
