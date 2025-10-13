# PowerShell script to reorganize agents into new folder structure

$baseDir = "C:\Users\psytz\.claude\agents"
Set-Location $baseDir

# Move orchestration agents
Write-Host "Moving orchestration agents..."
Move-Item -Path "orchestrators\*.md" -Destination "00-orchestration\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "09-meta-orchestration\*.md" -Destination "00-orchestration\" -Force -ErrorAction SilentlyContinue

# Move backend development agents
Write-Host "Moving backend development agents..."
Move-Item -Path "01-core-development\backend-developer.md" -Destination "01-development\backend\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "01-core-development\api-designer.md" -Destination "01-development\backend\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "01-core-development\graphql-architect.md" -Destination "01-development\backend\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "01-core-development\microservices-architect.md" -Destination "01-development\backend\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "01-core-development\websocket-engineer.md" -Destination "01-development\backend\" -Force -ErrorAction SilentlyContinue

# Move frontend development agents
Write-Host "Moving frontend development agents..."
Move-Item -Path "01-core-development\frontend-developer.md" -Destination "01-development\frontend\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "01-core-development\ui-designer.md" -Destination "01-development\frontend\" -Force -ErrorAction SilentlyContinue

# Move mobile development agents
Write-Host "Moving mobile development agents..."
Move-Item -Path "01-core-development\mobile-developer.md" -Destination "01-development\mobile\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "07-specialized-domains\mobile-app-developer.md" -Destination "01-development\mobile\" -Force -ErrorAction SilentlyContinue

# Move fullstack agents
Write-Host "Moving fullstack agents..."
Move-Item -Path "01-core-development\fullstack-developer.md" -Destination "01-development\fullstack\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "01-core-development\electron-pro.md" -Destination "01-development\fullstack\" -Force -ErrorAction SilentlyContinue

# Move language specialists
Write-Host "Moving language specialists..."
# Web languages
Move-Item -Path "02-language-specialists\javascript-pro.md" -Destination "02-languages\web\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "02-language-specialists\typescript-pro.md" -Destination "02-languages\web\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "02-language-specialists\php-pro.md" -Destination "02-languages\web\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "02-language-specialists\sql-pro.md" -Destination "02-languages\web\" -Force -ErrorAction SilentlyContinue

# Systems languages
Move-Item -Path "02-language-specialists\cpp-pro.md" -Destination "02-languages\systems\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "02-language-specialists\rust-engineer.md" -Destination "02-languages\systems\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "02-language-specialists\golang-pro.md" -Destination "02-languages\systems\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "02-language-specialists\java-architect.md" -Destination "02-languages\systems\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "02-language-specialists\csharp-developer.md" -Destination "02-languages\systems\" -Force -ErrorAction SilentlyContinue

# Scripting languages
Move-Item -Path "02-language-specialists\python-pro.md" -Destination "02-languages\scripting\" -Force -ErrorAction SilentlyContinue

# Move framework specialists
Write-Host "Moving framework specialists..."
# Frontend frameworks
Move-Item -Path "02-language-specialists\react-specialist.md" -Destination "03-frameworks\frontend\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "02-language-specialists\vue-expert.md" -Destination "03-frameworks\frontend\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "02-language-specialists\angular-architect.md" -Destination "03-frameworks\frontend\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "02-language-specialists\nextjs-developer.md" -Destination "03-frameworks\frontend\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "specialized\react\*.md" -Destination "03-frameworks\frontend\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "specialized\vue\*.md" -Destination "03-frameworks\frontend\" -Force -ErrorAction SilentlyContinue

# Backend frameworks
Move-Item -Path "02-language-specialists\django-developer.md" -Destination "03-frameworks\backend\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "02-language-specialists\rails-expert.md" -Destination "03-frameworks\backend\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "02-language-specialists\laravel-specialist.md" -Destination "03-frameworks\backend\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "02-language-specialists\spring-boot-engineer.md" -Destination "03-frameworks\backend\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "02-language-specialists\dotnet-core-expert.md" -Destination "03-frameworks\backend\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "02-language-specialists\dotnet-framework-4.8-expert.md" -Destination "03-frameworks\backend\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "specialized\django\*.md" -Destination "03-frameworks\backend\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "specialized\laravel\*.md" -Destination "03-frameworks\backend\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "specialized\rails\*.md" -Destination "03-frameworks\backend\" -Force -ErrorAction SilentlyContinue

# Mobile frameworks
Move-Item -Path "02-language-specialists\flutter-expert.md" -Destination "03-frameworks\mobile\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "02-language-specialists\swift-expert.md" -Destination "03-frameworks\mobile\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "02-language-specialists\kotlin-specialist.md" -Destination "03-frameworks\mobile\" -Force -ErrorAction SilentlyContinue

# Move infrastructure agents
Write-Host "Moving infrastructure agents..."
# Cloud
Move-Item -Path "03-infrastructure\cloud-architect.md" -Destination "04-infrastructure\cloud\" -Force -ErrorAction SilentlyContinue

# DevOps
Move-Item -Path "03-infrastructure\devops-engineer.md" -Destination "04-infrastructure\devops\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "03-infrastructure\deployment-engineer.md" -Destination "04-infrastructure\devops\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "03-infrastructure\platform-engineer.md" -Destination "04-infrastructure\devops\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "03-infrastructure\sre-engineer.md" -Destination "04-infrastructure\devops\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "03-infrastructure\kubernetes-specialist.md" -Destination "04-infrastructure\devops\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "03-infrastructure\terraform-engineer.md" -Destination "04-infrastructure\devops\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "03-infrastructure\devops-incident-responder.md" -Destination "04-infrastructure\devops\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "03-infrastructure\database-administrator.md" -Destination "04-infrastructure\devops\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "03-infrastructure\network-engineer.md" -Destination "04-infrastructure\devops\" -Force -ErrorAction SilentlyContinue

# Security
Move-Item -Path "03-infrastructure\security-engineer.md" -Destination "04-infrastructure\security\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "03-infrastructure\security-incident-responder.md" -Destination "04-infrastructure\security\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "04-quality-security\security-auditor.md" -Destination "04-infrastructure\security\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "04-quality-security\penetration-tester.md" -Destination "04-infrastructure\security\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "04-quality-security\compliance-auditor.md" -Destination "04-infrastructure\security\" -Force -ErrorAction SilentlyContinue

# Move quality agents
Write-Host "Moving quality agents..."
# Testing
Move-Item -Path "04-quality-security\qa-expert.md" -Destination "05-quality\testing\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "04-quality-security\test-automator.md" -Destination "05-quality\testing\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "04-quality-security\accessibility-tester.md" -Destination "05-quality\testing\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "04-quality-security\chaos-engineer.md" -Destination "05-quality\testing\" -Force -ErrorAction SilentlyContinue

# Review
Move-Item -Path "04-quality-security\code-reviewer.md" -Destination "05-quality\review\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "04-quality-security\architect-reviewer.md" -Destination "05-quality\review\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "core\code-archaeologist.md" -Destination "05-quality\review\" -Force -ErrorAction SilentlyContinue

# Performance
Move-Item -Path "04-quality-security\performance-engineer.md" -Destination "05-quality\performance\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "core\performance-optimizer.md" -Destination "05-quality\performance\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "04-quality-security\debugger.md" -Destination "05-quality\performance\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "04-quality-security\error-detective.md" -Destination "05-quality\performance\" -Force -ErrorAction SilentlyContinue

# Move data and AI agents
Write-Host "Moving data and AI agents..."
# Data
Move-Item -Path "05-data-ai\data-engineer.md" -Destination "06-data-ai\data\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "05-data-ai\data-scientist.md" -Destination "06-data-ai\data\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "05-data-ai\data-analyst.md" -Destination "06-data-ai\data\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "05-data-ai\database-optimizer.md" -Destination "06-data-ai\data\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "05-data-ai\postgres-pro.md" -Destination "06-data-ai\data\" -Force -ErrorAction SilentlyContinue

# ML
Move-Item -Path "05-data-ai\ml-engineer.md" -Destination "06-data-ai\ml\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "05-data-ai\mlops-engineer.md" -Destination "06-data-ai\ml\" -Force -ErrorAction SilentlyContinue

# AI
Move-Item -Path "05-data-ai\ai-engineer.md" -Destination "06-data-ai\ai\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "05-data-ai\llm-architect.md" -Destination "06-data-ai\ai\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "05-data-ai\nlp-engineer.md" -Destination "06-data-ai\ai\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "05-data-ai\prompt-engineer.md" -Destination "06-data-ai\ai\" -Force -ErrorAction SilentlyContinue

# Move specialized domain agents
Write-Host "Moving specialized domain agents..."
# Fintech
Move-Item -Path "07-specialized-domains\fintech-engineer.md" -Destination "07-specialized\fintech\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "07-specialized-domains\payment-integration.md" -Destination "07-specialized\fintech\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "07-specialized-domains\quant-analyst.md" -Destination "07-specialized\fintech\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "07-specialized-domains\risk-manager.md" -Destination "07-specialized\fintech\" -Force -ErrorAction SilentlyContinue

# Blockchain
Move-Item -Path "07-specialized-domains\blockchain-developer.md" -Destination "07-specialized\blockchain\" -Force -ErrorAction SilentlyContinue

# Gaming
Move-Item -Path "07-specialized-domains\game-developer.md" -Destination "07-specialized\gaming\" -Force -ErrorAction SilentlyContinue

# IoT
Move-Item -Path "07-specialized-domains\iot-engineer.md" -Destination "07-specialized\iot\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "07-specialized-domains\embedded-systems.md" -Destination "07-specialized\iot\" -Force -ErrorAction SilentlyContinue

# Move support agents
Write-Host "Moving support agents..."
# Documentation
Move-Item -Path "core\documentation-specialist.md" -Destination "08-support\documentation\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "06-developer-experience\documentation-engineer.md" -Destination "08-support\documentation\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "08-business-product\technical-writer.md" -Destination "08-support\documentation\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "07-specialized-domains\api-documenter.md" -Destination "08-support\documentation\" -Force -ErrorAction SilentlyContinue

# Management
Move-Item -Path "08-business-product\project-manager.md" -Destination "08-support\management\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "08-business-product\product-manager.md" -Destination "08-support\management\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "08-business-product\scrum-master.md" -Destination "08-support\management\" -Force -ErrorAction SilentlyContinue

# Business
Move-Item -Path "08-business-product\business-analyst.md" -Destination "08-support\business\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "08-business-product\sales-engineer.md" -Destination "08-support\business\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "08-business-product\customer-success-manager.md" -Destination "08-support\business\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "08-business-product\content-marketer.md" -Destination "08-support\business\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "08-business-product\legal-advisor.md" -Destination "08-support\business\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "08-business-product\ux-researcher.md" -Destination "08-support\business\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "08-business-product\wordpress-master.md" -Destination "08-support\business\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "07-specialized-domains\seo-specialist.md" -Destination "08-support\business\" -Force -ErrorAction SilentlyContinue

# Move utility agents
Write-Host "Moving utility agents..."
Move-Item -Path "06-developer-experience\*.md" -Destination "09-utilities\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "10-research-analysis\*.md" -Destination "09-utilities\" -Force -ErrorAction SilentlyContinue
Move-Item -Path "universal\*.md" -Destination "09-utilities\" -Force -ErrorAction SilentlyContinue

# Clean up old empty directories
Write-Host "Cleaning up old directories..."
Remove-Item -Path "01-core-development", "02-language-specialists", "03-infrastructure", "04-quality-security", "05-data-ai", "06-developer-experience", "07-specialized-domains", "08-business-product", "09-meta-orchestration", "10-research-analysis", "orchestrators", "core", "specialized", "universal" -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "Reorganization complete!"