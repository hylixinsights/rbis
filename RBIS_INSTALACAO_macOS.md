# 🍎 Guia de Instalação RBIS para macOS

*Versão 4.0 — Março 2026*

---

## ⚡ RESUMO RÁPIDO (para quem tem pressa)

```bash
# 1. Navegar até a pasta
cd ~/meus_projetos/rbis

# 2. Criar ambiente virtual
python3 -m venv rbis_env
source rbis_env/bin/activate

# 3. Instalar dependências + RBIS
pip install --upgrade pip setuptools wheel
pip install numpy scipy anndata pandas scanpy matplotlib seaborn numba joblib
pip install -e .

# 4. Verificar
python3 -c "import rbis; print('✓ RBIS pronto!')"
```

---

## 📋 PASSO A PASSO COMPLETO

### **PASSO 1: Verificar a Instalação do Python**

Abra o **Terminal** (Cmd + Espaço → "Terminal" → Enter):

```bash
python3 --version
pip3 --version
```

**Você deve ver algo como:**
```
Python 3.10.8 (ou superior)
pip 23.0.1
```

**Se não estiver instalado:**

#### Opção A: Via Homebrew (Recomendado)
```bash
# Instalar Homebrew (se não tiver)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Instalar Python 3
brew install python3
```

#### Opção B: Baixar direto
- Vá em https://www.python.org/downloads/
- Baixe a versão 3.10 ou superior
- Execute o instalador
- Siga as instruções (padrão é OK)

---

### **PASSO 2: Abrir o Terminal e Navegar**

```bash
# Abrir Terminal (Cmd + Espaço → "Terminal")

# Criar pasta para projetos (primeira vez apenas)
mkdir -p ~/meus_projetos

# Navegar para lá
cd ~/meus_projetos
```

---

### **PASSO 3: Obter o Código do RBIS**

#### Se está no GitHub:
```bash
git clone https://github.com/seu-usuario/rbis.git
cd rbis
```

(Se não tem `git`, instale com: `brew install git`)

#### Se tem arquivo `.zip`:
```bash
# Extrair manualmente no Finder
# Depois renomear a pasta para "rbis"

# No Terminal:
cd ~/meus_projetos/rbis
```

---

### **PASSO 4: Listar arquivos para confirmar estrutura**

```bash
ls -la
```

**Você deve ver:**
```
setup.py              ✓ (ou pyproject.toml)
README.md
rbis/
├── __init__.py
├── tl/
├── pl/
└── _core/
```

**Se não vir `setup.py`, veja a seção "Criar setup.py Manualmente" no final.**

---

### **PASSO 5: Criar Ambiente Virtual**

Um ambiente virtual isola as dependências do projeto:

```bash
# Criar o ambiente
python3 -m venv rbis_env

# Ativar o ambiente
source rbis_env/bin/activate
```

**Confirmação:** O prompt deve mudar para:
```bash
(rbis_env) usuario@MacBook-de-usuario:~/meus_projetos/rbis$
```

Se vir `(rbis_env)` no início, você está dentro do ambiente virtual. ✓

---

### **PASSO 6: Atualizar pip, setuptools e wheel**

```bash
pip install --upgrade pip setuptools wheel
```

Você verá algo como:
```
Successfully installed pip-23.0.1 setuptools-67.0.0 wheel-0.40.0
```

---

### **PASSO 7: Instalar Dependências Núcleo**

```bash
pip install numpy scipy anndata pandas scanpy matplotlib seaborn
```

Isto vai levar alguns minutos. Você verá linhas como:
```
Collecting numpy
Downloading numpy-1.24.0-cp310-cp310-macosx_11_0_arm64.whl
...
Successfully installed numpy-1.24.0 scipy-1.10.0 ... [muitas linhas]
```

**Esperado para macOS M1/M2:** Se vir avisos sobre `arm64` ou `universal2`, é normal e OK.

---

### **PASSO 8: Instalar Pacotes Opcionais (Recomendado)**

Para melhor performance:

```bash
pip install numba joblib
```

---

### **PASSO 9: Instalar o Pacote RBIS em Modo Desenvolvimento**

Este é o passo crucial:

```bash
pip install -e .
```

**Explicação:**
- `-e` = "editable mode" — qualquer mudança no código aparece imediatamente
- `.` = instala da pasta atual

**Você deve ver:**
```
Successfully installed rbis-4.0.0
```

Se tiver erro aqui, pulte para **TROUBLESHOOTING** no final.

---

### **PASSO 10: Verificar a Instalação**

Teste se tudo funcionou:

**Teste rápido 1:**
```bash
python3 -c "import rbis; print(f'✓ RBIS {rbis.__version__} instalado com sucesso!')"
```

**Teste rápido 2:**
```bash
python3 -c "import rbis.tl; import rbis.pl; print('✓ Módulos tl e pl carregados!')"
```

**Teste completo (Python interativo):**
```bash
python3
```

Depois dentro do prompt Python:
```python
>>> import rbis
>>> import rbis.tl
>>> import rbis.pl
>>> import scanpy as sc
>>> import anndata as ad
>>> print("✓ Todos os módulos carregados com sucesso!")
>>> exit()
```

Se tudo funcionar, parabéns! 🎉

---

## ✅ PRÓXIMOS PASSOS (Seu Primeiro Análise)

Crie um arquivo `teste_rbis.py`:

```bash
nano teste_rbis.py
```

Cole isto:

```python
import scanpy as sc
import rbis

# Se tiver um dataset
# adata = sc.read_h5ad('seu_arquivo.h5ad')

# Ou criar dados de teste
import numpy as np
import anndata as ad

# Dataset fictício: 1000 células, 5000 genes, 5 clusters
X = np.random.poisson(5, (1000, 5000))
clusters = np.repeat(['A', 'B', 'C', 'D', 'E'], 200)
adata = ad.AnnData(X=X, obs={'cluster': clusters}, var=ad.io.zarr.open_array(None))
adata.obs['cluster'] = clusters

print("Dataset carregado!")
print(f"Shape: {adata.shape}")
print(f"Clusters: {adata.obs['cluster'].unique()}")

# Executar RBIS
print("\n🔍 Executando RBIS...")
rbis.tl.find_markers_sc(adata, groupby='cluster', target_n=50, random_state=42)

# Ver resultados
print("\n📊 Cluster Report:")
print(adata.uns['rbis']['cluster_report'][['identity_score', 'n_signature_genes', 'confidence_flag']])

print("\n✓ Análise concluída!")
```

Salve (Ctrl+X → Y → Enter) e rode:

```bash
python3 teste_rbis.py
```

---

## 🐛 TROUBLESHOOTING para macOS

### **Erro: "No module named 'setup'"**

O arquivo `setup.py` está faltando. Crie manualmente (veja "Criar setup.py" abaixo).

### **Erro: "ModuleNotFoundError: No module named 'rbis'"**

1. Verifique se está no ambiente virtual:
   ```bash
   which python3
   # Deve mostrar: /Users/seu_usuario/meus_projetos/rbis/rbis_env/bin/python3
   ```

2. Se não estiver, reative:
   ```bash
   source rbis_env/bin/activate
   ```

3. Tente instalar novamente:
   ```bash
   pip install -e .
   ```

### **Erro: "Permission denied" ou "Cannot create directory"**

NÃO use `sudo`. Em vez disso:

```bash
# Remova qualquer instalação anterior
pip uninstall rbis -y

# Reinstale sem sudo
pip install -e .
```

### **Erro: "No such file or directory: setup.py"**

Você não está na pasta correta. Verifique:

```bash
pwd
# Deve mostrar: /Users/seu_usuario/meus_projetos/rbis

ls setup.py
# Se não existir, crie (veja abaixo)
```

### **Erro: "ImportError: numpy not installed"**

Instale novamente as dependências:

```bash
pip install --upgrade pip
pip install numpy scipy anndata pandas scanpy
pip install -e .
```

### **Aviso: "Requires: python_requires >=3.8, but installing for 3.7"**

Seu Python é muito velho. Atualize:

```bash
brew install python3
```

### **Aviso: "File name too long" (macOS M1/M2)**

Raramente ocorre. Se acontecer, reinicie o Terminal.

### **Erro ao importar scanpy: "ImportError: cannot import name '_plot_params'"**

Versões incompatíveis. Atualize:

```bash
pip install --upgrade scanpy anndata
```

---

## 🔧 CRIAR setup.py MANUALMENTE

Se não tem `setup.py`, crie:

```bash
nano setup.py
```

Cole isto:

```python
from setuptools import setup, find_packages

setup(
    name="rbis",
    version="4.0.0",
    description="Rank-Based Identity Search for robust biomarker discovery",
    author="Seu Nome",
    author_email="seu.email@example.com",
    url="https://github.com/seu-usuario/rbis",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "anndata>=0.8.0",
        "scanpy>=1.9.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
    ],
    extras_require={
        "dev": [
            "numba>=0.55.0",
            "joblib>=1.1.0",
        ],
        "test": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
```

Salve (Ctrl+X → Y → Enter) e rode:

```bash
pip install -e .
```

---

## 💡 DICAS EXTRAS PARA macOS

### Usar VS Code para editar código

```bash
# Instalar VS Code (se não tiver)
brew install visual-studio-code

# Abrir pasta no VS Code
code .
```

### Usar Jupyter Notebook

```bash
pip install jupyter

# Iniciar
jupyter notebook

# Será aberto em http://localhost:8888
```

### Manter o ambiente virtual salvo

Para não perder o ambiente, coloque ele em `.gitignore`:

```bash
echo "rbis_env/" > .gitignore
```

### Desativar o ambiente virtual

```bash
deactivate
```

O prompt volta ao normal.

### Reativar o ambiente depois

```bash
source rbis_env/bin/activate
```

### Reinstalar tudo do zero (se algo der muito errado)

```bash
# Desativar ambiente
deactivate

# Deletar ambiente
rm -rf rbis_env

# Criar novo
python3 -m venv rbis_env
source rbis_env/bin/activate

# Instalar tudo novamente
pip install --upgrade pip setuptools wheel
pip install numpy scipy anndata pandas scanpy matplotlib seaborn numba joblib
pip install -e .
```

---

## 📚 PRÓXIMOS RECURSOS

Quando o RBIS estiver funcionando:

1. **Documentação completa:** Veja `RBIS_Vignette_v4_0_EN.md`
2. **Exemplos de código:** Seção 8 (Quickstart)
3. **Troubleshooting científico:** Seção 11 (Common Failure Modes)
4. **API completa:** Seções 9 e 10 (Input Parameters, Module Dependency)

---

## ✨ CHECKLIST FINAL

Antes de usar RBIS, confirme:

- [ ] Python 3.10+ instalado (`python3 --version`)
- [ ] Ambiente virtual criado (`source rbis_env/bin/activate`)
- [ ] Dependências instaladas (pip mostra todas as libs)
- [ ] RBIS instalado (`python3 -c "import rbis"` funciona)
- [ ] Você consegue rodar `python3` interativamente
- [ ] Você consegue sair de Python (digitando `exit()`)

Se tudo verde, você está pronto! 🚀

---

**Qualquer dúvida, envie a mensagem de erro exato que recebeu. Estarei aqui para ajudar!**
