# Meu Projeto

Este é um projeto incrível que faz coisas incríveis.

## Configuração do Ambiente de Desenvolvimento

### 1. Instalando o Visual Studio Code

Certifique-se de ter o Visual Studio Code instalado em sua máquina. Você pode baixá-lo em [Visual Studio Code](https://code.visualstudio.com/).

### 2. Configurando um Ambiente Virtual

Abra o terminal no Visual Studio Code e siga estes passos para criar um ambiente virtual:

```bash
# Instalar a extensão do Python (se ainda não estiver instalada)
# Certifique-se de ter o Python devidamente instalado na sua máquina.
# Clique em "View" -> "Extensions" ou pressione `Ctrl+Shift+X`, procure por "Python" e instale a extensão.

# Criar um diretório para o ambiente virtual
mkdir venv

# Navegar até o diretório do ambiente virtual
cd venv

# Criar um ambiente virtual (substitua "venv" pelo nome que preferir)
python3 -m venv venv

# Ativar o ambiente virtual
# No Linux ou macOS
source venv/bin/activate

# No Windows
venv\Scripts\activate


# Ativando o ambiente virtual (se ainda não estiver ativado)
# No Linux ou macOS
source venv/bin/activate

# No Windows
venv\Scripts\activate

# Instalando o Python 3.7
python3.7 -m pip install --upgrade pip


# Certifique-se de estar no diretório do projeto e com o ambiente virtual ativado

# Instalando as dependências
pip install -r requirements.txt
