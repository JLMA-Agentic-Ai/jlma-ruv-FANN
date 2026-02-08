//! Kubernetes / NKE deployment manifest generation for cuda-wasm workloads
//!
//! Generates complete Kubernetes YAML manifests for deploying cuda-wasm GPU workloads
//! on Nutanix Kubernetes Engine (NKE) clusters, including:
//!
//! - Deployments with GPU resource requests (NVIDIA, AMD)
//! - Node affinity rules for GPU vendor selection
//! - ConfigMaps for cuda-wasm runtime configuration
//! - PersistentVolumeClaims for kernel cache (Nutanix CSI)
//! - Services and HorizontalPodAutoscalers
//! - NKE-specific annotations and labels

use super::config::*;
use std::collections::HashMap;

/// Generates Kubernetes deployment manifests for cuda-wasm workloads
pub struct DeploymentGenerator {
    config: DeploymentConfig,
}

impl DeploymentGenerator {
    /// Create a new DeploymentGenerator from deployment configuration
    pub fn new(config: DeploymentConfig) -> Self {
        Self { config }
    }

    /// Generate a complete set of Kubernetes manifests as a single multi-document YAML string
    ///
    /// The output includes (separated by `---`):
    /// 1. Namespace
    /// 2. ConfigMap for runtime settings
    /// 3. PersistentVolumeClaim for kernel cache
    /// 4. Deployment with GPU resource requests
    /// 5. Service
    /// 6. HorizontalPodAutoscaler (if enabled)
    pub fn generate_all(&self) -> String {
        let mut manifests = vec![
            self.generate_namespace(),
            self.generate_configmap(),
            self.generate_pvc(),
            self.generate_deployment(),
            self.generate_service(),
        ];

        if self.config.enable_hpa {
            manifests.push(self.generate_hpa());
        }

        manifests.join("\n---\n")
    }

    /// Generate the Namespace manifest
    pub fn generate_namespace(&self) -> String {
        format!(
            r#"apiVersion: v1
kind: Namespace
metadata:
  name: {namespace}
  labels:
    app.kubernetes.io/part-of: cuda-wasm
    platform: nutanix-nke"#,
            namespace = self.config.namespace
        )
    }

    /// Generate the ConfigMap for cuda-wasm runtime settings
    pub fn generate_configmap(&self) -> String {
        let mut env_entries = String::new();
        for (key, value) in &self.config.env_vars {
            env_entries.push_str(&format!("    {}={}\n", key, value));
        }

        let gpu_backend = match &self.config.gpu_vendor {
            GpuVendor::Nvidia => "cuda",
            GpuVendor::Amd => "rocm",
            GpuVendor::Intel => "oneapi",
            GpuVendor::Unknown(_) => "webgpu",
        };

        format!(
            r#"apiVersion: v1
kind: ConfigMap
metadata:
  name: {name}-config
  namespace: {namespace}
  labels:
    app.kubernetes.io/name: {name}
    app.kubernetes.io/component: config
data:
  CUDA_WASM_GPU_BACKEND: "{gpu_backend}"
  CUDA_WASM_GPU_COUNT: "{gpu_count}"
  CUDA_WASM_KERNEL_CACHE_DIR: "/cache/kernels"
  CUDA_WASM_LOG_LEVEL: "info"
  CUDA_WASM_WEBGPU_ENABLED: "true"
  CUDA_WASM_MEMORY_POOL_SIZE: "2147483648"
  CUDA_WASM_MAX_CONCURRENT_KERNELS: "16"
{env_entries}"#,
            name = self.config.name,
            namespace = self.config.namespace,
            gpu_backend = gpu_backend,
            gpu_count = self.config.gpus_per_pod,
            env_entries = if env_entries.is_empty() {
                String::new()
            } else {
                format!("  # Custom environment variables\n{}", env_entries)
            }
        )
    }

    /// Generate the PersistentVolumeClaim for kernel cache storage (Nutanix CSI)
    pub fn generate_pvc(&self) -> String {
        format!(
            r#"apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: {name}-kernel-cache
  namespace: {namespace}
  labels:
    app.kubernetes.io/name: {name}
    app.kubernetes.io/component: cache
  annotations:
    # Nutanix CSI volume annotations
    csi.nutanix.com/storage-type: "NutanixVolumes"
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: {storage_class}
  resources:
    requests:
      storage: {cache_size}"#,
            name = self.config.name,
            namespace = self.config.namespace,
            storage_class = self.config.storage_class,
            cache_size = self.config.kernel_cache_size
        )
    }

    /// Generate the Deployment manifest with GPU resource requests and node affinity
    pub fn generate_deployment(&self) -> String {
        let gpu_resource = gpu_resource_key(&self.config.gpu_vendor);
        let labels = self.merge_labels();
        let annotations = self.merge_annotations();

        let labels_yaml = format_yaml_map(&labels, 8);
        let annotations_yaml = format_yaml_map(&annotations, 8);
        let selector_labels = format!(
            "app.kubernetes.io/name: {}\n        app.kubernetes.io/instance: {}",
            self.config.name, self.config.name
        );
        let pod_labels_yaml = format_yaml_map(&labels, 12);

        let node_affinity = self.generate_node_affinity();
        let tolerations = self.generate_tolerations();

        format!(
            r#"apiVersion: apps/v1
kind: Deployment
metadata:
  name: {name}
  namespace: {namespace}
  labels:
{labels_yaml}
  annotations:
{annotations_yaml}
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      {selector_labels}
  template:
    metadata:
      labels:
{pod_labels_yaml}
    spec:
{node_affinity}
{tolerations}
      containers:
        - name: cuda-wasm-worker
          image: {image}
          ports:
            - containerPort: {port}
              name: http
              protocol: TCP
          envFrom:
            - configMapRef:
                name: {name}-config
          resources:
            requests:
              cpu: "{cpu_request}"
              memory: "{mem_request}"
              {gpu_resource}: "{gpu_count}"
            limits:
              cpu: "{cpu_limit}"
              memory: "{mem_limit}"
              {gpu_resource}: "{gpu_count}"
          volumeMounts:
            - name: kernel-cache
              mountPath: /cache/kernels
            - name: dshm
              mountPath: /dev/shm
          livenessProbe:
            httpGet:
              path: /healthz
              port: http
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /readyz
              port: http
            initialDelaySeconds: 10
            periodSeconds: 5
      volumes:
        - name: kernel-cache
          persistentVolumeClaim:
            claimName: {name}-kernel-cache
        - name: dshm
          emptyDir:
            medium: Memory
            sizeLimit: 8Gi"#,
            name = self.config.name,
            namespace = self.config.namespace,
            replicas = self.config.replicas,
            image = self.config.image,
            port = self.config.service_port,
            cpu_request = self.config.cpu_request,
            cpu_limit = self.config.cpu_limit,
            mem_request = self.config.memory_request,
            mem_limit = self.config.memory_limit,
            gpu_resource = gpu_resource,
            gpu_count = self.config.gpus_per_pod,
            labels_yaml = labels_yaml,
            annotations_yaml = annotations_yaml,
            selector_labels = selector_labels,
            pod_labels_yaml = pod_labels_yaml,
            node_affinity = node_affinity,
            tolerations = tolerations,
        )
    }

    /// Generate the Service manifest
    pub fn generate_service(&self) -> String {
        format!(
            r#"apiVersion: v1
kind: Service
metadata:
  name: {name}
  namespace: {namespace}
  labels:
    app.kubernetes.io/name: {name}
    app.kubernetes.io/component: api
spec:
  type: ClusterIP
  ports:
    - port: {port}
      targetPort: http
      protocol: TCP
      name: http
  selector:
    app.kubernetes.io/name: {name}
    app.kubernetes.io/instance: {name}"#,
            name = self.config.name,
            namespace = self.config.namespace,
            port = self.config.service_port
        )
    }

    /// Generate the HorizontalPodAutoscaler manifest
    pub fn generate_hpa(&self) -> String {
        let gpu_resource = gpu_resource_key(&self.config.gpu_vendor);

        format!(
            r#"apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {name}-hpa
  namespace: {namespace}
  labels:
    app.kubernetes.io/name: {name}
    app.kubernetes.io/component: autoscaler
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {name}
  minReplicas: {min}
  maxReplicas: {max}
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 80
    - type: Pods
      pods:
        metric:
          name: {gpu_resource}_utilization
        target:
          type: AverageValue
          averageValue: "{target_util}"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
        - type: Pods
          value: 2
          periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Pods
          value: 1
          periodSeconds: 120"#,
            name = self.config.name,
            namespace = self.config.namespace,
            min = self.config.hpa_min_replicas,
            max = self.config.hpa_max_replicas,
            gpu_resource = gpu_resource.replace('/', "_"),
            target_util = self.config.hpa_target_gpu_utilization,
        )
    }

    // --- Private helpers ---

    /// Merge default labels with user-supplied labels
    fn merge_labels(&self) -> HashMap<String, String> {
        let mut labels = HashMap::new();
        labels.insert(
            "app.kubernetes.io/name".to_string(),
            self.config.name.clone(),
        );
        labels.insert(
            "app.kubernetes.io/instance".to_string(),
            self.config.name.clone(),
        );
        labels.insert(
            "app.kubernetes.io/component".to_string(),
            "gpu-worker".to_string(),
        );
        labels.insert(
            "app.kubernetes.io/part-of".to_string(),
            "cuda-wasm".to_string(),
        );
        labels.insert(
            "app.kubernetes.io/managed-by".to_string(),
            "cuda-wasm-deployer".to_string(),
        );

        // Add GPU vendor label
        let vendor_label = match &self.config.gpu_vendor {
            GpuVendor::Nvidia => "nvidia",
            GpuVendor::Amd => "amd",
            GpuVendor::Intel => "intel",
            GpuVendor::Unknown(v) => v.as_str(),
        };
        labels.insert("cuda-wasm/gpu-vendor".to_string(), vendor_label.to_string());

        // Merge user labels
        for (k, v) in &self.config.labels {
            labels.insert(k.clone(), v.clone());
        }

        labels
    }

    /// Merge default annotations with user-supplied and NKE-specific annotations
    fn merge_annotations(&self) -> HashMap<String, String> {
        let mut annotations = HashMap::new();

        // NKE-specific annotations
        annotations.insert(
            "nke.nutanix.com/gpu-enabled".to_string(),
            "true".to_string(),
        );
        annotations.insert(
            "nke.nutanix.com/cluster-type".to_string(),
            "gpu-workload".to_string(),
        );

        // Merge user annotations
        for (k, v) in &self.config.annotations {
            annotations.insert(k.clone(), v.clone());
        }

        annotations
    }

    /// Generate node affinity rules for GPU vendor selection
    fn generate_node_affinity(&self) -> String {
        let vendor_label_value = match &self.config.gpu_vendor {
            GpuVendor::Nvidia => "nvidia",
            GpuVendor::Amd => "amd",
            GpuVendor::Intel => "intel",
            GpuVendor::Unknown(v) => v.as_str(),
        };

        format!(
            r#"      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - key: nvidia.com/gpu.present
                    operator: In
                    values:
                      - "true"
                  - key: feature.node.kubernetes.io/pci-{vendor}.present
                    operator: In
                    values:
                      - "true"
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              preference:
                matchExpressions:
                  - key: cuda-wasm/gpu-vendor
                    operator: In
                    values:
                      - "{vendor}""#,
            vendor = vendor_label_value,
        )
    }

    /// Generate tolerations for GPU nodes
    fn generate_tolerations(&self) -> String {
        r#"      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
        - key: amd.com/gpu
          operator: Exists
          effect: NoSchedule
        - key: "node-role.kubernetes.io/gpu"
          operator: Exists
          effect: NoSchedule"#
            .to_string()
    }
}

/// Get the Kubernetes GPU resource key for a given vendor
pub fn gpu_resource_key(vendor: &GpuVendor) -> &'static str {
    match vendor {
        GpuVendor::Nvidia => "nvidia.com/gpu",
        GpuVendor::Amd => "amd.com/gpu",
        GpuVendor::Intel => "gpu.intel.com/i915",
        GpuVendor::Unknown(_) => "nvidia.com/gpu", // default to NVIDIA
    }
}

/// Format a HashMap as indented YAML key-value pairs
fn format_yaml_map(map: &HashMap<String, String>, indent: usize) -> String {
    let prefix = " ".repeat(indent);
    let mut pairs: Vec<_> = map.iter().collect();
    pairs.sort_by_key(|(k, _)| k.clone());

    pairs
        .iter()
        .map(|(k, v)| format!("{}{}: \"{}\"", prefix, k, v))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> DeploymentConfig {
        DeploymentConfig::new("test-workload", "cuda-wasm:v1.0")
            .with_gpu_vendor(GpuVendor::Nvidia)
            .with_gpus(2)
            .with_hpa(1, 4, 75)
    }

    #[test]
    fn test_generate_namespace() {
        let gen = DeploymentGenerator::new(test_config());
        let yaml = gen.generate_namespace();
        assert!(yaml.contains("kind: Namespace"));
        assert!(yaml.contains("name: cuda-wasm"));
    }

    #[test]
    fn test_generate_configmap() {
        let gen = DeploymentGenerator::new(test_config());
        let yaml = gen.generate_configmap();
        assert!(yaml.contains("kind: ConfigMap"));
        assert!(yaml.contains("CUDA_WASM_GPU_BACKEND: \"cuda\""));
        assert!(yaml.contains("CUDA_WASM_GPU_COUNT: \"2\""));
    }

    #[test]
    fn test_generate_pvc() {
        let gen = DeploymentGenerator::new(test_config());
        let yaml = gen.generate_pvc();
        assert!(yaml.contains("kind: PersistentVolumeClaim"));
        assert!(yaml.contains("storageClassName: nutanix-volume"));
        assert!(yaml.contains("csi.nutanix.com/storage-type"));
    }

    #[test]
    fn test_generate_deployment_nvidia() {
        let gen = DeploymentGenerator::new(test_config());
        let yaml = gen.generate_deployment();
        assert!(yaml.contains("kind: Deployment"));
        assert!(yaml.contains("nvidia.com/gpu: \"2\""));
        assert!(yaml.contains("image: cuda-wasm:v1.0"));
        assert!(yaml.contains("nke.nutanix.com/gpu-enabled"));
    }

    #[test]
    fn test_generate_deployment_amd() {
        let config = DeploymentConfig::new("amd-workload", "cuda-wasm:v1.0")
            .with_gpu_vendor(GpuVendor::Amd);
        let gen = DeploymentGenerator::new(config);
        let yaml = gen.generate_deployment();
        assert!(yaml.contains("amd.com/gpu: \"1\""));
    }

    #[test]
    fn test_generate_service() {
        let gen = DeploymentGenerator::new(test_config());
        let yaml = gen.generate_service();
        assert!(yaml.contains("kind: Service"));
        assert!(yaml.contains("port: 8080"));
    }

    #[test]
    fn test_generate_hpa() {
        let gen = DeploymentGenerator::new(test_config());
        let yaml = gen.generate_hpa();
        assert!(yaml.contains("kind: HorizontalPodAutoscaler"));
        assert!(yaml.contains("minReplicas: 1"));
        assert!(yaml.contains("maxReplicas: 4"));
    }

    #[test]
    fn test_generate_all() {
        let gen = DeploymentGenerator::new(test_config());
        let yaml = gen.generate_all();
        // All sections should be present
        assert!(yaml.contains("kind: Namespace"));
        assert!(yaml.contains("kind: ConfigMap"));
        assert!(yaml.contains("kind: PersistentVolumeClaim"));
        assert!(yaml.contains("kind: Deployment"));
        assert!(yaml.contains("kind: Service"));
        assert!(yaml.contains("kind: HorizontalPodAutoscaler"));
        // Sections separated by ---
        assert!(yaml.matches("---").count() >= 5);
    }

    #[test]
    fn test_gpu_resource_key() {
        assert_eq!(gpu_resource_key(&GpuVendor::Nvidia), "nvidia.com/gpu");
        assert_eq!(gpu_resource_key(&GpuVendor::Amd), "amd.com/gpu");
        assert_eq!(gpu_resource_key(&GpuVendor::Intel), "gpu.intel.com/i915");
    }
}
