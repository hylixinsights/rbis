#!/bin/bash

###############################################################################
# RBIS → GitHub: Script Automatizado Completo
# Este script configura Git e faz o push para o GitHub de uma só vez
# Uso: bash setup_rbis_github.sh
###############################################################################

set -e  # Sair se houver erro

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  RBIS → GitHub: Setup Automatizado${NC}"
echo -e "${BLUE}========================================${NC}\n"

# ============================================================================
# PASSO 1: COLETAR INFORMAÇÕES DO USUÁRIO
# ============================================================================

echo -e "${YELLOW}PASSO 1: Informações Necessárias${NC}\n"

# Nome e Email
read -p "Seu nome completo: " GIT_NAME
read -p "Seu email GitHub: " GIT_EMAIL

# Caminho do projeto
read -p "Caminho completo do projeto RBIS (ex: /Users/Helder/Desktop/2026/Papers/RBIS): " PROJECT_PATH

# Remover trailing slash se houver
PROJECT_PATH="${PROJECT_PATH%/}"

# Validar que a pasta existe
if [ ! -d "$PROJECT_PATH" ]; then
    echo -e "${RED}❌ Erro: Pasta não encontrada em $PROJECT_PATH${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Pasta encontrada: $PROJECT_PATH${NC}\n"

# URL do repositório
read -p "URL do repositório GitHub (https://github.com/usuario/repositorio): " GITHUB_URL

# Método de autenticação
echo -e "\nMétodo de autenticação:"
echo "1) Personal Access Token (mais simples, recomendado)"
echo "2) SSH Key"
read -p "Escolha (1 ou 2): " AUTH_METHOD

case $AUTH_METHOD in
    1)
        AUTH_TYPE="token"
        read -p "Cole seu Personal Access Token do GitHub: " GITHUB_TOKEN
        ;;
    2)
        AUTH_TYPE="ssh"
        read -p "Caminho da sua chave SSH (ex: /Users/Helder/.ssh/id_ed25519): " SSH_KEY
        if [ ! -f "$SSH_KEY" ]; then
            echo -e "${RED}❌ Erro: Arquivo de chave SSH não encontrado${NC}"
            exit 1
        fi
        ;;
    *)
        echo -e "${RED}❌ Opção inválida${NC}"
        exit 1
        ;;
esac

echo ""

# ============================================================================
# PASSO 2: VALIDAR GIT INSTALADO
# ============================================================================

echo -e "${YELLOW}PASSO 2: Verificando Git${NC}\n"

if ! command -v git &> /dev/null; then
    echo -e "${RED}❌ Git não instalado. Instalando via Homebrew...${NC}"
    if ! command -v brew &> /dev/null; then
        echo -e "${RED}❌ Homebrew também não está instalado.${NC}"
        echo "Instale Homebrew em: https://brew.sh"
        exit 1
    fi
    brew install git
fi

GIT_VERSION=$(git --version)
echo -e "${GREEN}✓ Git instalado: $GIT_VERSION${NC}\n"

# ============================================================================
# PASSO 3: CONFIGURAR GIT GLOBALMENTE
# ============================================================================

echo -e "${YELLOW}PASSO 3: Configurando Git${NC}\n"

git config --global user.name "$GIT_NAME"
git config --global user.email "$GIT_EMAIL"

echo -e "${GREEN}✓ Configurado:${NC}"
echo "  Nome: $(git config --global user.name)"
echo "  Email: $(git config --global user.email)"
echo ""

# ============================================================================
# PASSO 4: PREPARAR PASTA DO PROJETO
# ============================================================================

echo -e "${YELLOW}PASSO 4: Preparando Pasta do Projeto${NC}\n"

cd "$PROJECT_PATH"

# Listar arquivos
echo "Arquivos encontrados:"
ls -la | head -20
echo ""

# ============================================================================
# PASSO 5: CRIAR ARQUIVOS ESSENCIAIS (se não existem)
# ============================================================================

echo -e "${YELLOW}PASSO 5: Criando Arquivos Essenciais${NC}\n"

# .gitignore
if [ ! -f ".gitignore" ]; then
    echo "Criando .gitignore..."
    cat > .gitignore << 'EOF'
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
venv/
env/
rbis_env/
.venv
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
*.h5ad
*.csv
*.xlsx
*.parquet
data/
results/
.ipynb_checkpoints/
*.ipynb
.coverage
htmlcov/
*.log
.env
.cache/
EOF
    echo -e "${GREEN}✓ .gitignore criado${NC}"
else
    echo -e "${GREEN}✓ .gitignore já existe${NC}"
fi

# LICENSE
if [ ! -f "LICENSE" ]; then
    echo "Criando LICENSE..."
    cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
EOF
    echo -e "${GREEN}✓ LICENSE criado${NC}"
else
    echo -e "${GREEN}✓ LICENSE já existe${NC}"
fi

# README.md
if [ ! -f "README.md" ]; then
    echo "Criando README.md..."
    cat > README.md << 'EOF'
# RBIS — Rank-Based Identity Search

A consensus, exclusion, and dynamic-margin framework for robust biomarker discovery.

## Features

- **Rank-based marker discovery** — Identifies genes that best define cluster identity
- **Two-layer ranking for sparsity** — Optimized for single-cell data with dropout
- **Dynamic margin sieve** — Gene-specific exclusivity thresholds
- **Negative marker discovery** — Detects genes specifically silenced in clusters
- **Cross-fidelity mapping** — Identifies transitional and mixed-identity cells
- **Permutation validation** — Empirical FDR testing for cluster structure
- **Comprehensive diagnostics** — Cell/sample fidelity scoring and quality checks

## Quick Start

```python
import scanpy as sc
import rbis

adata = sc.read_h5ad('dataset.h5ad')
adata_full = adata.raw.to_adata()
adata_full.obs['leiden'] = adata.obs['leiden']

rbis.tl.find_markers_sc(adata_full, groupby='leiden', target_n=100)
rbis.tl.find_silenced_sc(adata_full, groupby='leiden', target_n_neg=50)

print(adata_full.uns['rbis']['cluster_report'])
rbis.pl.specificity_climb(adata_full)
```

## Installation

```bash
pip install -e .
```

## Documentation

See [RBIS_Vignette_v4_0_EN.md](./RBIS_Vignette_v4_0_EN.md) for complete documentation.

## License

MIT License — see [LICENSE](./LICENSE) for details.
EOF
    echo -e "${GREEN}✓ README.md criado${NC}"
else
    echo -e "${GREEN}✓ README.md já existe${NC}"
fi

echo ""

# ============================================================================
# PASSO 6: INICIALIZAR GIT
# ============================================================================

echo -e "${YELLOW}PASSO 6: Inicializando Git${NC}\n"

if [ -d ".git" ]; then
    echo -e "${GREEN}✓ Repositório Git já existe${NC}"
else
    git init
    echo -e "${GREEN}✓ Repositório Git inicializado${NC}"
fi

# Adicionar arquivos
echo "Adicionando arquivos ao Git..."
git add .
echo -e "${GREEN}✓ Arquivos adicionados${NC}\n"

# ============================================================================
# PASSO 7: FAZER COMMIT
# ============================================================================

echo -e "${YELLOW}PASSO 7: Criando Commit${NC}\n"

# Verificar se há mudanças
if git diff --cached --quiet; then
    echo -e "${YELLOW}⚠ Nenhuma mudança para commitar${NC}"
else
    git commit -m "Initial commit: RBIS v4.0.0 - Rank-Based Identity Search"
    echo -e "${GREEN}✓ Commit criado${NC}\n"
fi

# ============================================================================
# PASSO 8: CONFIGURAR REMOTE E BRANCH
# ============================================================================

echo -e "${YELLOW}PASSO 8: Configurando Remote do GitHub${NC}\n"

# Remover origin antigo se existir
if git remote get-url origin &> /dev/null; then
    echo "Removendo origin antigo..."
    git remote remove origin
fi

# Adaptar URL se necessário para autenticação
if [ "$AUTH_TYPE" = "token" ]; then
    # Adicionar token à URL se usando token
    GITHUB_URL_WITH_AUTH=$(echo "$GITHUB_URL" | sed "s|https://|https://$GIT_EMAIL:$GITHUB_TOKEN@|")
    git remote add origin "$GITHUB_URL_WITH_AUTH"
else
    # Converter para SSH se necessário
    GITHUB_URL_SSH=$(echo "$GITHUB_URL" | sed 's|https://github.com/|git@github.com:|' | sed 's|/$||')
    git remote add origin "$GITHUB_URL_SSH"
fi

echo -e "${GREEN}✓ Remote configurado${NC}\n"

# ============================================================================
# PASSO 9: RENOMEAR BRANCH PARA MAIN
# ============================================================================

echo -e "${YELLOW}PASSO 9: Renomeando Branch para 'main'${NC}\n"

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CURRENT_BRANCH" != "main" ]; then
    git branch -M main
    echo -e "${GREEN}✓ Branch renomeado para 'main'${NC}"
else
    echo -e "${GREEN}✓ Branch já é 'main'${NC}"
fi
echo ""

# ============================================================================
# PASSO 10: FAZER PUSH
# ============================================================================

echo -e "${YELLOW}PASSO 10: Fazendo Push para GitHub${NC}\n"
echo "Isto pode levar alguns segundos..."

if git push -u origin main; then
    echo -e "${GREEN}✓ Push realizado com sucesso!${NC}\n"
else
    echo -e "${RED}❌ Erro ao fazer push${NC}"
    echo "Verifique:"
    echo "  1. A URL do repositório está correta"
    echo "  2. Seu token/chave SSH está válido"
    echo "  3. Você tem permissão para fazer push"
    exit 1
fi

# ============================================================================
# RESUMO FINAL
# ============================================================================

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✓ SETUP COMPLETO!${NC}"
echo -e "${BLUE}========================================${NC}\n"

echo -e "${GREEN}Resumo do que foi feito:${NC}"
echo "  ✓ Git configurado localmente"
echo "  ✓ Arquivos essenciais criados"
echo "  ✓ Repositório inicializado"
echo "  ✓ Remote GitHub configurado"
echo "  ✓ Arquivos commitados"
echo "  ✓ Push realizado\n"

echo -e "${BLUE}Próximos passos:${NC}"
echo "  1. Visite: $GITHUB_URL"
echo "  2. Verifique que todos os arquivos estão lá"
echo "  3. Configure webhooks/CI-CD conforme necessário\n"

# Verificar status
echo -e "${YELLOW}Status final:${NC}\n"
git status
echo ""
git log --oneline -5
echo ""

echo -e "${GREEN}Tudo pronto! 🚀${NC}"
