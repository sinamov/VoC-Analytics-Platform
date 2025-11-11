// -----------------------------------------------------------------------------
// Target Scope: This file is intended to be deployed at the Resource Group scope.
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// 1. PARAMETERS
// Parameters are inputs we provide at deployment time.
// -----------------------------------------------------------------------------

// 'location' will automatically use the same location as the Resource Group 
// you deploy this file into. This is a best practice.
param location string = resourceGroup().location

// 'baseName' will be the unique prefix for all your resources.
// We'll generate a unique string based on the resource group's ID
// to ensure all resource names are globally unique.
param baseName string = 'voc${uniqueString(resourceGroup().id)}'

// -----------------------------------------------------------------------------
// 2. VARIABLES
// Variables are used to construct names and settings inside this file.
// -----------------------------------------------------------------------------

var storageAccountName = '${baseName}storage'
var databricksWorkspaceName = '${baseName}-dbw'
var containerRegistryName = '${baseName}acr'
var openAiServiceName = '${baseName}-oai'
var openAiDeploymentName = 'gpt-35-turbo' // The specific model we want to use

// -----------------------------------------------------------------------------
// 3. RESOURCES
// This is the core of the file, defining the 5 resources you planned.
// -----------------------------------------------------------------------------

// --- Resource 1: Azure Blob Storage (for .mbox files) ---
resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: storageAccountName
  location: location
  sku: {
    // Standard_LRS is the cheapest, perfect for this project.
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    accessTier: 'Hot'
    supportsHttpsTrafficOnly: true
    isHnsEnabled: false // 'isHnsEnabled: true' would make it a Data Lake,
                        // but simple Blob Storage is fine for this.
    allowBlobPublicAccess: false
  }
}

// --- Resource 2: Azure Databricks (for Spark job) ---
// This is the most complex resource.
resource databricksWorkspace 'Microsoft.Databricks/workspaces@2023-02-01' = {
  name: databricksWorkspaceName
  location: location
  sku: {
    // Per your plan, we'll use a premium SKU to start.
    // For your "Cost-Saving Note": You can change this to 'standard'
    // or 'trial' (if eligible) and re-run the deployment.
    // 'premium' is often required for more advanced features.
    name: 'premium'
  }
  properties: {
    // Databricks needs its own managed resource group for its clusters.
    // This constructs a full, valid resource group ID for it.
    managedResourceGroupId: '${subscription().id}/resourceGroups/managed-rg-${databricksWorkspaceName}'
  }
}

// --- Resource 3: Azure Container Registry (for Docker image) ---
resource containerRegistry 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: containerRegistryName
  location: location
  sku: {
    // 'Basic' is a great, low-cost SKU for a portfolio project.
    name: 'Basic'
  }
  properties: {
    // We enable the admin user for simplicity. In a real CI/CD
    // pipeline, you would use a more secure Service Principal.
    adminUserEnabled: true
  }
}

// --- Resource 4: Azure OpenAI (for gpt-3.5-turbo) ---
resource openAiService 'Microsoft.CognitiveServices/accounts@2023-05-01' = {
  name: openAiServiceName
  location: location
  sku: {
    // 'S0' is the standard, pay-as-you-go SKU.
    name: 'S0'
  }
  kind: 'OpenAI'
  properties: {
    // This is important: We must accept the terms.
    customSubDomainName: openAiServiceName
    publicNetworkAccess: 'Enabled'
  }

  // This 'deployment' sub-resource creates the *actual model endpoint*.
  // We are nesting it inside the 'openAiService' resource.
  // This 'deployment' sub-resource creates the *actual model endpoint*.
  // We'll deploy a common embedding model just to get the service created.
  // ... inside the openAiService resource
resource modelDeployment 'deployments@2023-05-01' = { // <-- Match the parent version
  name: 'text-embedding-ada' 
  // ... rest of the resource
    sku: {
      name: 'Standard'
      capacity: 30 // 30,000 Tokens-Per-Minute
    }
    properties: {
      model: {
        format: 'OpenAI'
        name: 'text-embedding-ada-002' // New model name
        version: '2'                    // New, stable version
      }
    }
  }
}

// -----------------------------------------------------------------------------
// 4. OUTPUTS
// Outputs will print important values to the terminal after deployment,
// like API keys and hostnames, so we can use them later.
// -----------------------------------------------------------------------------

output OPENAI_ENDPOINT string = openAiService.properties.endpoint
output OPENAI_SERVICE_NAME string = openAiService.name
output ACR_LOGIN_SERVER string = containerRegistry.properties.loginServer
output DATABRICKS_URL string = databricksWorkspace.properties.workspaceUrl
output STORAGE_ACCOUNT_NAME string = storageAccount.name
