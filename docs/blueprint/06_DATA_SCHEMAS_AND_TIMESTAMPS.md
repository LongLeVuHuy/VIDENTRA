# 06 — Data schemas, frame indexing và timestamp contracts

**Normative v1.0.** Các ID và khoảng thời gian không được tự suy ra từ nominal FPS. [Pipeline§13.2/13.11/13.13](05_VIDENTRA_DETAILED_PIPELINE.md) dùng hợp đồng này.

## 1. Identity và coordinate systems

| Field | Type/meaning | Invariant |
|---|---|---|
| video_id | stable string, một ingested asset version | không đổi khi index rebuild |
| source_video_id | stable upstream/source identity | nhiều clips có thể cùng source; source URI không bắt buộc accessible |
| media_sha256 | hex64 content hash | rename không đổi bytes identity; identical bytes có alias metadata khác |
| sampled_frame_id | UUID derived namespace(video_id,timeline_revision,ordinal,sampler_revision) | khác original ordinal và codec I-frame |
| original_decoded_frame_index | int64, zero-based ordinal của **all successfully decoded presentation-order frames** trong ingested video | tăng tuần tự từ0; pin decoder/stream/error policy |
| source_original_frame_index | int64 nullable | chỉ có khi ingest/ledger full source và frame equivalence verified |
| vector_id | positive signed int64; JSON serialize string nếu >2^53−1 | registry cấp, never reuse; independent of FAISS row positions |
| evidence_id/event_id/track_id | stable unique string trong asset+recipe namespace | traceable support refs, không share identity qua cut ngầm |
| snapshot_id | hash/version của complete serving manifest | request pin một snapshot |

“Original decoded” không phải camera capture ID bí mật. Nếu file hỏng khiến decoder bỏ frame thì ordinal theo declared decoder output, không khẳng định biết index của frame không decode được. Gaps/corruption làm authoritative frame-export cần review; không sửa bằng fps. Sample representative frame không đồng nghĩa codec keyframe/I-frame. Codec flag chỉ giúp seek.

## 2. Time basis

Authoritative observation time `p_i = PTS_i × time_base_num / time_base_den` (rational seconds). Lưu cả integer PTS lẫn rational time_base; DTS là decode order, không dùng để playback moment. `best_effort_timestamp` riêng, không ghi đè missing raw PTS.

Chọn origin `o` là first valid video presentation time cho asset có video. Audio-only asset dùng audio first time. `t_clip = p_i − o`. Audio timestamps cũng trừ **cùng o**, giữ possible negative audio preroll và audio-delay. Playback proxy có map riêng, không reset audio/video về zero độc lập.

Constant-offset source mapping chỉ khi cut rate1 đã xác minh:
`t_source = source_offset + t_clip`.
General piecewise map row: `clip_start_us, clip_end_us, source_start_us, rate_num, rate_den, source_epoch_id, mapping_provenance`; `t_source = source_start + (t_clip − clip_start)×rate_num/rate_den`. Discontinuous edit/cut có nhiều rows. Unknown source offset→null, không parse filename rồi coi authoritative; filename chỉ hint cần provenance.

Storage search joins: signed integer microseconds, nearest rounding half-to-even từ rational; raw rational vẫn authoritative. JSON `timestamp_seconds` là derived display numeric, không dùng làm unique key. Frame PTS trùng dùng `pts_occurrence` + ordinal; unstable/backward time split epoch; không join across epoch thiếu explicit map.

### Worked timeline — SYNTHETIC

Video firstPTS=1200, time_base1/1000, audio firstPTS1550 cùng time_base: video clip0, audio clip0,35s. Frame representative rawPTS15600→clip14,4s. Source offset210s→source224,4s. ASR chunk bắt đầu clip15s, word relative0,2s→clip15,2s→source225,2s. Không snap15,2 vào nearest visual sample.

VFR PTS [1200,1240,1330,1340]→clip [0,0,04,0,13,0,14] với ordinals[0,1,2,3]. t=0,13 không suy ra index3 từ25fps; ledger trả index2. Frame instant khác duration support: first frame at0 là observation tại0, không tuyên bố có cùng semantics tới0,04 nếu chưa xem.

## 3. Moments, boundaries và frame endpoints

Moment/event/ASR interval dùng half-open [start_us,end_us), end>start. OCR singleton dùng `timestamp_us`, không [t,t] giả interval. Tracked OCR có observation list, first/last, observed support segments và gap uncertainty; persistence không vượt last observation nếu không evidence tiếp theo.

Predicted moment boundary có `method, uncertainty_start_us, uncertainty_end_us, resolution_us, coarse`. Human GT uncertainty tách khỏi model uncertainty. Start-frame = first frame PTS≥start; end-frame = last PTS<end, **inclusive ordinal**. Start/end seconds vẫn giữ prediction boundaries hoặc snapped boundaries theo explicit method, không đổi ngầm evaluator. Nếu no frame exists thì reject visual moment hoặc nullable frame IDs audio-only, nêu reason.

Proxy map có `proxy_asset_id, proxy_pts/timebase, clip_time_us, source_frame_index` hoặc piecewise timestamp mapping. Browser currentTime chỉ approximate display; exact frame-step từ frame endpoint decode source. VFR stepping dùng adjacent ordinal trong ledger, không currentTime+=1/fps. ASR word bounds có thể ở giữa frames, bình thường.

## 4. Machine-readable schemas

- [MediaAsset](../schemas/media_asset.schema.json): source/hash/streams/origin/rights/split/exposure/modality status.
- [FrameRecord](../schemas/frame_record.schema.json): mọi keyframe metadata requested, original ordinal/PTS/timebase/source offset/scene/reasons/dimensions/fingerprint.
- [QueryGraph](../schemas/query_graph.schema.json): đầy đủ schema cũng in tại pipeline§13.8.
- [Search result envelope item](../schemas/result.schema.json): required output fields; typed evidence subrecords theo bảng§5; [complete synthetic result](../schemas/result.synthetic.json).
- [Việt example](../schemas/query_vi.synthetic.json) và [English example](../schemas/query_en.synthetic.json).

JSON Schema kiểm type/shape. Semantic validators phải kiểm nonempty interval, source transform math, referential integrity, result evidence IDs và localize relation refs. Result schema intentionally cho phép evidence payload theo `kind`; M0 phải compile kind-specific validators theo§5, không coi arbitrary JSON payload đủ validation. Provenance.synthetic=false bắt buộc SHA-256 real media; synthetic exceptions không được vào production serving.

## 5. Evidence records: exact field contract

Mọi evidence record có:
`evidence_id:string, video_id:string, source_video_id:string, epoch_id:string, timeline_revision:string, kind:enum, recipe_hash:hex64, model_id:string|null, model_revision:string|null, input_hashes:string[], generator_type:model|rule|human, created_at:RFC3339, raw_scores:object<string,number>, score_semantics:string, quality_flags:string[], status:ready|failed|missing|not_run|empty_observed`.
Các common fields được store archival; API có thể trả projection nhưng phải có provenance ref đầy đủ accessible.

| Kind | Required payload | Null/constraints |
|---|---|---|
| FrameObservation | frame_id,original_decoded_frame_index,timestamp_us,dimensions,width/height,fingerprint | timestamp null chỉ quarantined; ordinal phải exist |
| OCRObservation | frame_id,timestamp_us,polygon_px:[x,y]×4+,text_original,text_nfc,language_hypotheses,detector_revision,recognizer_revision,crop_hash | polygon source-image bounds; no invented end timestamp |
| OCRTrack | observation_ids[],first_observed_us,last_observed_us,association_method,association_scores[],holes[],persistence_kind,boundary_uncertainty_us | ≥2 observations để tracked_interval; singleton persistence_kind=instant |
| ASRSegment | start_us,end_us,text_original,text_nfc,language,chunk_id,audio_sample_start,audio_sample_end,timestamp_method,word_records[],speaker_cluster|null | no audio-visual identity implied; optional words may null timings |
| EntityTrack | track_id,entity_type,frame_boxes:[frame_id,bbox,score][],attribute_observations[],track_links[],identity_confidence|null | new segment after cut; track_links uncertainty explicit |
| EventHypothesis | event_id,predicate,roles:map<role,track_id>,start_us,end_us,support_refs[],assessment_state,method,boundary_uncertainty | one frame/keyword insufficient for handover; role refs validated |
| RelationAssessment | query_predicate_id,event_or_track_refs[],state:supported|contradicted|unknown,features,reason_codes[],support_refs[],contradict_refs[] | unknown has no fabricated evidence refs |
| SpeakerBinding | speaker_cluster,actor_track_id,start_us,end_us,audio_visual_model_revision,support_frame_refs[],sync_error_us,state | acoustic diarization alone cannot supported actor binding |
| CandidateMoment | candidate_id,seed_branch_ids[],video_id,epoch_id,start_us,end_us,proposal_method,seed_evidence_ids[],original_query_rank|null | proposal region not event truth |
| FinalMoment | candidate plus rank features,final boundaries,frame endpoints,representative,evidence,match_status,score_type | pipeline result contract; rank differs from confidence |

Confidence fields store raw, per-model calibrated optional separate object `{value,calibrator_revision,calibration_split,scope}`. Never overwrite raw confidence. Recognizer alternatives on same crop share group; evidence counts should not pretend independence.

Human annotations dùng separate `GroundTruthRecord`:
`annotation_id, dataset_revision, official_split, query_id, source_video_id, positive_moments[], negatives[], unjudged[], annotation_type, creation_method, reviewer_receipts[], adjudication_status, exposure, coordinate_system, source_hash, label_hash`.
Không copy GT vào production evidence/FTS tables. Human review form trống giữ pending.

## 6. Metadata DDL tối thiểu (đặc tả)

```sql
PRAGMA foreign_keys=ON;
CREATE TABLE media(
 video_id TEXT PRIMARY KEY,
 source_video_id TEXT NOT NULL,
 sha256 TEXT NOT NULL,
 media_uri TEXT NOT NULL,
 timeline_revision TEXT NOT NULL,
 rights_status TEXT NOT NULL CHECK(rights_status IN ('allowed','restricted','unknown'))
);
CREATE TABLE entity_registry(
 entity_id TEXT PRIMARY KEY,
 video_id TEXT NOT NULL REFERENCES media(video_id),
 kind TEXT NOT NULL,
 timeline_locator TEXT NOT NULL
);
CREATE TABLE vector_registry(
 vector_id INTEGER PRIMARY KEY AUTOINCREMENT,
 entity_id TEXT NOT NULL REFERENCES entity_registry(entity_id),
 encoder_revision TEXT NOT NULL,
 representation_kind TEXT NOT NULL,
 tombstoned INTEGER NOT NULL DEFAULT 0 CHECK(tombstoned IN (0,1)),
 UNIQUE(entity_id,encoder_revision,representation_kind)
);
CREATE TABLE evidence_locator(
 evidence_id TEXT PRIMARY KEY,
 video_id TEXT NOT NULL REFERENCES media(video_id),
 epoch_id TEXT NOT NULL,
 kind TEXT NOT NULL,
 start_us INTEGER,
 end_us INTEGER,
 timestamp_us INTEGER,
 payload_uri TEXT NOT NULL,
 payload_sha256 TEXT NOT NULL,
 CHECK((timestamp_us IS NOT NULL AND start_us IS NULL AND end_us IS NULL)
    OR (timestamp_us IS NULL AND start_us IS NOT NULL
        AND end_us IS NOT NULL AND end_us>start_us))
);
CREATE INDEX evidence_by_time ON evidence_locator(video_id,epoch_id,start_us,end_us);
CREATE INDEX evidence_by_instant ON evidence_locator(video_id,epoch_id,timestamp_us);
CREATE VIRTUAL TABLE ocr_fts USING fts5(
 evidence_id UNINDEXED, text_nfc, text_folded,
 tokenize='unicode61 remove_diacritics 0'
);
CREATE VIRTUAL TABLE asr_fts USING fts5(
 evidence_id UNINDEXED, text_nfc, text_folded,
 tokenize='unicode61 remove_diacritics 0'
);
CREATE TABLE snapshots(
 snapshot_id TEXT PRIMARY KEY,
 manifest_uri TEXT NOT NULL,
 manifest_sha256 TEXT NOT NULL,
 state TEXT NOT NULL CHECK(state IN ('building','validated','active','retired'))
);
```

Evidence summary nodes không có point/interval (ví dụ query-only relation) không chèn vào evidence_locator: nằm typed payload referenced bởi event/candidate. OCR track singleton phải locator instant. Sidecar shards + SQLite durable registry là source of truth; FAISS chỉ index vectors.

FTS insertion transaction cùng locator version, snapshot build hoàn tất rồi publish. Rebuild copy mapping existing IDs; AUTOINCREMENT uniqueness được coordinator quản lý, offline shard workers không tự cấp trùng. Branch snapshots include shard IDs/codebook hashes/nprobe config; score behavior versioned.

## 7. Job và consistency contracts

Job: `job_id,asset_hash,stage_id,recipe_hash,attempt,status,lease_until,input_hashes,output_hashes,error_code,retryable,started_at,completed_at`. State pending→running→validated→committed; failed/retry transitions giữ attempts. Lease expiry cho worker recovery; only validated output ready. Rerun cache hit phải verify manifest, không skip dựa thư mục tồn tại.

Index manifest: `snapshot_id,schema_version,encoder_specs,shards[],vector_registry_hash,timeline_catalog_hash,evidence_catalog_hash,text_index_hash,rights_manifest_hash,publish_time`. Query result ghi chính snapshot. New media can publish visual-ready with OCR/ASR not_run flags; capability-complete status riêng.

## 8. Acceptance details

E01/E09 fixtures: CFR, true VFR, nonzero positive/negative starts, repeated PTS, B-frame reorder, audio leading/trailing, source cut offsets, speed edit piecewise, corrupt packet, no audio, empty stream. Verify rational→clip→source→inverse within one microsecond rounding; exact ledger frame hashes match direct decoding. Snapshot rebuild IDs unchanged; invalid mix versions rejected.

Không tuyên bố timestamps chính xác hơn model/human annotation uncertainty. Frame mapping có thể exact đối với predicted interval nhưng prediction boundary vẫn sai; báo hai lỗi riêng.
