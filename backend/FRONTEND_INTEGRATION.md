# Подключение frontend teammate

Backend запускается **на том же компьютере, что и ваш браузер**. `127.0.0.1` на другом компьютере указывает на другой компьютер; этот этап не является общим публичным сервером.

Базовый URL: `http://127.0.0.1:8000`. Дизайн, маршрутизация и компоненты frontend остаются на вашей стороне. Ключ OpenAI в frontend не нужен.

## Минимальный маршрут

1. `GET /api/health`: `status=ok` означает, что сервер работает. `llm_configured=true` означает только наличие ключа, не проверку его действительности.
2. `POST /api/conversations/import`: отправить исходный JSON организаторов или `{ "records": [...] }`. Пример — `examples/import.json`.
3. `GET /api/conversations`: прочитать массив `items`, вывести список. `status=not_evaluated` относится к диалогу без запусков.
4. `GET /api/checklists`: показать версии и выбрать hash. `items[0]` — последняя сохранённая версия, которая может быть пользовательской. В каждой записи доступны `criteria`, `title`, `version`.
5. `POST /api/evaluations`: отправить ID разговора и hash правил.
6. Опросить `GET /api/evaluations/{evaluation_id}` раз в секунду до конечного состояния.
7. Показать `result.final_findings`, `result.final_summary` и исходные `messages`. После перезагрузки получить историю через карточку диалога и открыть нужный запуск по ID.

## Запуск оценки

```json
{
  "conversation_id": "synthetic-example-en",
  "checklist_hash": "hash_из_GET_api_checklists",
  "force": false
}
```

`checklist_hash` в настоящем запросе — строка из 64 hex-символов, возвращённая сервером, не заглушка из примера.

Ответ HTTP **202** означает queued/running; **200** означает возврат сохранённого результата. Оба ответа имеют одинаковые поля:

```json
{
  "evaluation_id": "идентификатор",
  "conversation_id": "synthetic-example-en",
  "checklist_hash": "тот_же_hash",
  "status": "queued",
  "result": null,
  "error": null,
  "inference": { "requested_model": "gpt-6.1-sol", "calls": 0, "usage": {} },
  "created_at": "ISO 8601",
  "updated_at": "ISO 8601"
}
```

Параметры модели определяет backend. Клиент не может передавать ключ, адрес OpenAI или произвольную модель в запросе оценки.

## Пример JavaScript

```javascript
const API = 'http://127.0.0.1:8000';
const terminal = new Set(['completed', 'needs_review', 'error']);

async function api(path, body) {
  const response = await fetch(`${API}/api${path}`, body === undefined ? {} : {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const value = await response.json();
  if (!response.ok) throw new Error(value.error?.message ?? `HTTP ${response.status}`);
  return value;
}

async function evaluateConversation(conversationId, checklistHash, onUpdate, force = false) {
  let evaluation = await api('/evaluations', {
    conversation_id: conversationId,
    checklist_hash: checklistHash,
    force,
  });
  onUpdate(evaluation);
  while (!terminal.has(evaluation.status)) {
    await new Promise(resolve => setTimeout(resolve, 1000));
    evaluation = await api(`/evaluations/${encodeURIComponent(evaluation.evaluation_id)}`);
    onUpdate(evaluation);
  }
  return evaluation;
}
```

При сетевой ошибке polling можно повторить GET с тем же `evaluation_id`. Не отправляйте новый платный запуск вместо проверки состояния. Повторная отправка обычного POST также защищена кэшем, но `force:true` после завершения создаёт новую попытку.

## Что отображать

| Поле/состояние | Поведение frontend |
|---|---|
| `queued` / `running` | Индикатор обработки; продолжать GET polling |
| `completed` | Показать сохранённую оценку; остановить polling |
| `needs_review` | Показать доступные выводы и ограничения; остановить polling |
| `error` | Показать `error.message`; `result=null`; не показывать нулевой или успешный балл |
| `result.final_summary.score` | Основной итог. `null` → «Итог не рассчитывается» |
| `partial_score_on_assessed` | Только диагностика; не использовать как общий балл |
| `final_summary.critical_error=true` | Обнаружено грубое нарушение |
| `critical_error=false` | Нарушение не отмечено по завершённой проверке флага |
| `critical_error=null` | Флаг не определён; не писать «Нарушений нет» |

Критерий из `final_findings` содержит `criterion_id`, `status` (`scored`, `na`, `needs_review`), `score`, `answer`, `reason`, `evidence`, `checked_span`, `na_reason`, `verification_limits`, `factual_status` и `knowledge_evidence`.

Название, инструкция, вес и шкала находятся в `result.checklist_snapshot.criteria`: соединять по `id === criterion_id`, **не по позиции в массиве**. Критерии c01–c07 дают баллы; c08 — отдельный флаг, где «Да» означает нарушение. Итоги и вклады берутся из `final_summary`, не пересчитываются браузером.

Это описание официального начального чек-листа. После редактирования ID и число критериев могут быть другими: форму и карточки строить по снимку конфигурации. `type=flag` означает отдельный флаг, `weight=0`; доступные ступени — `allowed_scores`, применимость — `allow_na`. При редактировании копируйте конфигурацию, без служебного поля `hash`.

## Новая версия чек-листа

Отправьте `POST /api/checklists`:

```javascript
const saved = await api('/checklists', {
  checklist: editedChecklist,
  parent_hash: selectedHash,
  author: 'teammate',
});
// HTTP 201: новая версия; 200: идентичное содержимое уже есть.
// Для следующих оценок используйте saved.hash.
```

`GET /api/checklists/{hash}` возвращает конкретную версию. Поддерживаются добавление/удаление критерия, формулировки, инструкции, веса, `binary|scale|flag`, минимум/максимум/ступени, N/A и секции. Невалидная конфигурация — 422. Все версии неизменяемы; изменение правил не пересчитывает историческую оценку автоматически. Новый запуск использует выбранный hash. Правило округления текущего backend — HALF_UP.

## Апелляция и решение

Автор и проверяющий — поля аудита демо, без регистрации. Создать обращение можно для `completed`/`needs_review` с сохранённым результатом:

```javascript
const appeal = await api(`/evaluations/${evaluationId}/appeals`, {
  criterion_id: criterionId,
  comment: 'Почему оценку нужно пересмотреть',
  author: 'Оператор',
  client_request_id: crypto.randomUUID(),
});
```

ID запроса создаётся один раз для действия и сохраняется при сетевом повторе. Повтор с тем же содержимым не создаёт второе обращение; другое содержимое с тем же ID — 409. Вторую открытую апелляцию на тот же критерий сервер отклоняет. Создание обращения не меняет балл.

`GET /api/evaluations/{id}/appeals` → `{items, revision}`. Каждое обращение содержит исходный оспоренный вывод, комментарий, автора, состояние и, после решения, проверяющего/причину/замену. Полная карточка оценки также содержит `result.appeals`, `revision`, `review_status=none|open|resolved`.

Перед решением получите свежий `GET /api/evaluations/{id}`. Для подтверждения:

```javascript
const resolved = await api(`/appeals/${appealId}/resolve`, {
  decision: 'uphold',
  reviewer: 'Проверяющий',
  reason: 'Оценка подтверждена после проверки реплик',
  replacement: null,
  expected_revision: currentEvaluation.revision,
  client_request_id: crypto.randomUUID(),
});
```

Для изменения передайте `decision:'change'` и `replacement` в формате finding: ID критерия, статус/балл, основание, точные цитаты/диапазон и причина N/A при необходимости. Пример — `examples/appeal.resolve.json`. Выбор ступени строится по `result.checklist_snapshot`; вместо неизвестности нельзя поставить произвольный ноль или запрещённый N/A.

Сервер проверит замену, атомарно пересчитает итог и вернёт полную оценку. `original_findings`/`original_summary` не меняются. Финальную карточку обновить из ответа. При `409 stale_revision` получить свежую карточку и показать изменение проверяющему; не перезаписывать автоматически. Повтор идентичного решения с тем же ID не применяет его повторно.

## Этапы LLM

`pipeline=single|cascade` задаётся сервером. `stage` может быть `queued`, `primary`, `routing`, `review`, `aggregation`, `finished`. Для промежуточного состояния достаточно сообщения «Оцениваем диалог»/«Проверяем оценку». `stages` содержит ответы и метаданные этапов, включая сохранённый первый этап при ошибке второго. Служебные промпты и параметры не требуется выводить пользователю.

`original_findings` — результат всей автоматической цепочки до человека, не обязательно ответ маленькой модели. Пока `status=running`, основной итог не показывать из промежуточного этапа. При `error` финального результата нет. `disagreements` и `routing` доступны для диагностической панели; они не подменяют ручное решение.

Доказательство: `{message_id, quote, speaker, timestamp}`. Найдите сообщение с тем же `message_id` в `result.messages`, прокрутите и подсветите. `checked_span` — `{from,to}`, обе границы включены. Для пропущенного действия показывайте проверенный участок; для `na` — причину N/A.

ID `m0001` уникален внутри конкретного диалога, не глобально. В DOM используйте составной ключ разговора/сообщения. Отображайте текст и цитаты как текст (например, React interpolation/textContent), не через HTML-вставку.

Временные метки исходные: MM:SS у чата, ISO 8601 с часовым поясом у звонка. Плеер можно подключить к `audio_url` карточки; точное соответствие календарных меток позиции в записи не предоставлено, переход к произвольной секунде по ним не вычисляйте.

## Ошибки и повторные запуски

Ошибки самого HTTP-запроса: `{ "error": { "code": "…", "message": "…" } }` с кодом 404/409/422/503. Валидация также может содержать `error.fields` без исходных пользовательских значений.

Ошибка фоновой оценки приходит в успешном GET HTTP 200, но `status=error`. Примеры кодов: `provider_auth`, `provider_rate_limit`, `provider_request`, `provider_timeout`, `provider_unavailable`, `model_refusal`, `incomplete_response`, `invalid_model_response`, `interrupted`.

После исправления причины предложите явное действие «Повторить» и отправьте `force:true`. Это новый вызов модели, предыдущая оценка остаётся в истории. Автоматического повтора нет. История: `GET /api/conversations/{id}`, поле `evaluations`; полный результат каждого запуска — отдельным GET по `evaluation_id`.

## CORS и запуск

Frontend на портах 3000/5173 (localhost или 127.0.0.1) разрешён по умолчанию. При другом порте добавить точный origin в `CQA_ALLOWED_ORIGINS` и перезапустить backend. Не используйте `credentials:'include'`: сессий и cookies здесь нет.

Точные схемы запросов доступны в `/docs` и поставляемом `openapi.json`. Готовые JSON-примеры успешной оценки, ручной проверки и ошибки лежат в `examples/`; все они синтетические и не являются реальными AI-результатами.
