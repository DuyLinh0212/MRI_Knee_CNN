import { Component, computed, OnDestroy, OnInit, signal } from '@angular/core';

type PlaneKey = 'axial' | 'coronal' | 'sagittal';
type TaskKey = 'abnormal' | 'acl' | 'meniscus';
type ModelKey = 'efficientnetb0' | 'densenet121' | 'efficientnetb0_vit';

interface ModelStatus {
  task: TaskKey;
  label: string;
  available: boolean;
  path: string | null;
  error: string | null;
}

interface HealthResponse {
  status: string;
  device: string;
  defaultModel?: ModelKey;
  imageSize: number;
  targetSlices: number;
  maxFilesPerPlane?: number;
  models: ModelStatus[];
  modelOptions?: ModelOption[];
}

interface ModelOption {
  name: ModelKey;
  label: string;
  availableCount: number;
  tasks: ModelStatus[];
}

interface TaskResult {
  label: string;
  available: boolean;
  prediction: boolean | null;
  probability: number | null;
  rawProbability?: number | null;
  weightPath: string | null;
  message: string;
  inferred?: boolean;
  inferredFrom?: TaskKey[];
}

interface PredictResponse {
  diagnosis: string;
  modelName?: ModelKey;
  modelLabel?: string;
  threshold: number;
  device: string;
  availableModels: TaskKey[];
  tasks: Record<TaskKey, TaskResult>;
}

interface PlanePreview {
  name: string;
  url: string;
}

@Component({
  selector: 'app-root',
  imports: [],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App implements OnInit, OnDestroy {
  protected readonly apiBaseUrl = 'http://127.0.0.1:8000';
  protected readonly maxFilesPerPlane = 30;
  protected readonly planeKeys: PlaneKey[] = ['axial', 'coronal', 'sagittal'];
  protected readonly taskOrder: TaskKey[] = ['abnormal', 'acl', 'meniscus'];
  protected readonly planeConfigs: Array<{ key: PlaneKey; title: string; subtitle: string }> = [
    { key: 'axial', title: 'Axial', subtitle: 'Mặt cắt ngang' },
    { key: 'coronal', title: 'Coronal', subtitle: 'Mặt cắt đứng trục' },
    { key: 'sagittal', title: 'Sagittal', subtitle: 'Mặt cắt dọc giữa' },
  ];

  protected readonly files = signal<Record<PlaneKey, File[]>>({
    axial: [],
    coronal: [],
    sagittal: [],
  });
  protected readonly previews = signal<Record<PlaneKey, PlanePreview[]>>({
    axial: [],
    coronal: [],
    sagittal: [],
  });
  protected readonly health = signal<HealthResponse | null>(null);
  protected readonly result = signal<PredictResponse | null>(null);
  protected readonly loading = signal(false);
  protected readonly statusLoading = signal(false);
  protected readonly error = signal<string | null>(null);
  protected readonly threshold = signal(0.5);
  protected readonly selectedModel = signal<ModelKey>('efficientnetb0');

  protected readonly canDiagnose = computed(
    () => this.planeKeys.every((plane) => this.files()[plane].length > 0) && !this.loading(),
  );
  protected readonly availableModelCount = computed(
    () => this.selectedModelOption()?.availableCount ?? this.health()?.models.filter((model) => model.available).length ?? 0,
  );
  protected readonly modelOptions = computed(() => this.health()?.modelOptions ?? []);
  protected readonly selectedModelOption = computed(
    () => this.modelOptions().find((model) => model.name === this.selectedModel()) ?? null,
  );
  protected readonly selectedModelTasks = computed(
    () => this.selectedModelOption()?.tasks ?? this.health()?.models ?? [],
  );
  protected readonly resultTasks = computed(() => {
    const tasks = this.result()?.tasks;
    if (!tasks) {
      return [];
    }

    return this.taskOrder.map((task) => ({ key: task, ...tasks[task] }));
  });

  ngOnInit(): void {
    void this.loadStatus();
  }

  ngOnDestroy(): void {
    for (const plane of this.planeKeys) {
      this.revokePlanePreviews(plane);
    }
  }

  protected async loadStatus(): Promise<void> {
    this.statusLoading.set(true);
    try {
      const response = await fetch(`${this.apiBaseUrl}/health`);
      if (!response.ok) {
        throw new Error(await this.readApiError(response));
      }
      const payload = (await response.json()) as HealthResponse;
      this.health.set(payload);
      if (payload.defaultModel && !this.selectedModelOption()) {
        this.selectedModel.set(payload.defaultModel);
      }
    } catch (error) {
      this.error.set(error instanceof Error ? error.message : 'Không thể kết nối Python API.');
    } finally {
      this.statusLoading.set(false);
    }
  }

  protected onFilesSelected(plane: PlaneKey, event: Event): void {
    const input = event.target as HTMLInputElement;
    const selectedFiles = Array.from(input.files ?? []);
    const acceptedFiles = selectedFiles.slice(0, this.maxFilesPerPlane);
    this.revokePlanePreviews(plane);
    this.files.update((current) => ({ ...current, [plane]: acceptedFiles }));
    this.previews.update((current) => ({ ...current, [plane]: this.createPreviews(acceptedFiles) }));
    this.result.set(null);
    this.error.set(
      selectedFiles.length > this.maxFilesPerPlane
        ? `Mỗi mặt cắt chỉ nhận tối đa ${this.maxFilesPerPlane} file. Hệ thống đã giữ ${this.maxFilesPerPlane} file đầu tiên.`
        : null,
    );
    input.value = '';
  }

  protected clearPlane(plane: PlaneKey): void {
    this.revokePlanePreviews(plane);
    this.files.update((current) => ({ ...current, [plane]: [] }));
    this.previews.update((current) => ({ ...current, [plane]: [] }));
    this.result.set(null);
  }

  protected onThresholdInput(event: Event): void {
    const input = event.target as HTMLInputElement;
    this.threshold.set(Number(input.value));
  }

  protected onModelChange(event: Event): void {
    const select = event.target as HTMLSelectElement;
    this.selectedModel.set(select.value as ModelKey);
    this.result.set(null);
  }

  protected getPlaneFiles(plane: PlaneKey): File[] {
    return this.files()[plane];
  }

  protected getPlanePreviews(plane: PlaneKey): PlanePreview[] {
    return this.previews()[plane];
  }

  protected fileSummary(plane: PlaneKey): string {
    const files = this.files()[plane];
    if (!files.length) {
      return 'Chưa chọn file';
    }

    const totalBytes = files.reduce((sum, file) => sum + file.size, 0);
    return `${files.length}/${this.maxFilesPerPlane} file - ${this.formatBytes(totalBytes)}`;
  }

  protected formatPercent(value: number | null): string {
    if (value === null) {
      return '--';
    }
    return `${(value * 100).toFixed(1)}%`;
  }

  protected async diagnose(): Promise<void> {
    if (!this.canDiagnose()) {
      this.error.set('Cần chọn đủ 3 mặt cắt axial, coronal và sagittal.');
      return;
    }

    this.loading.set(true);
    this.error.set(null);
    this.result.set(null);

    const formData = new FormData();
    for (const plane of this.planeKeys) {
      for (const file of this.files()[plane]) {
        formData.append(plane, file, file.name);
      }
    }
    formData.append('threshold', String(this.threshold()));
    formData.append('modelName', this.selectedModel());

    try {
      const response = await fetch(`${this.apiBaseUrl}/predict`, {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        throw new Error(await this.readApiError(response));
      }
      this.result.set((await response.json()) as PredictResponse);
      await this.loadStatus();
    } catch (error) {
      this.error.set(error instanceof Error ? error.message : 'Không thể chẩn đoán.');
    } finally {
      this.loading.set(false);
    }
  }

  private formatBytes(bytes: number): string {
    if (bytes < 1024) {
      return `${bytes} B`;
    }
    if (bytes < 1024 * 1024) {
      return `${(bytes / 1024).toFixed(1)} KB`;
    }
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  private async readApiError(response: Response): Promise<string> {
    try {
      const payload = await response.json();
      if (typeof payload.detail === 'string') {
        return payload.detail;
      }
      return JSON.stringify(payload.detail ?? payload);
    } catch {
      return `${response.status} ${response.statusText}`;
    }
  }

  private createPreviews(files: File[]): PlanePreview[] {
    return files
      .filter((file) => this.isBrowserPreviewableImage(file))
      .slice(0, this.maxFilesPerPlane)
      .map((file) => ({
        name: file.name,
        url: URL.createObjectURL(file),
      }));
  }

  private isBrowserPreviewableImage(file: File): boolean {
    if (/^image\/(png|jpeg|bmp|gif|webp)$/.test(file.type)) {
      return true;
    }

    return /\.(png|jpe?g|bmp|gif|webp)$/i.test(file.name);
  }

  private revokePlanePreviews(plane: PlaneKey): void {
    for (const preview of this.previews()[plane]) {
      URL.revokeObjectURL(preview.url);
    }
  }
}
