"""Execute review/edit JavaScript with a small DOM double (no GPU/browser)."""
from pathlib import Path
import shutil
import subprocess
import unittest


@unittest.skipUnless(shutil.which("node"), "Node is not installed")
class ReviewJavaScriptTests(unittest.TestCase):
    def test_pending_filter_optional_certification_and_edit_save(self):
        source = Path(__file__).resolve().parents[1] / "static/app.js"
        script = r"""
const fs = require('fs'), vm = require('vm'), assert = require('assert');
class Element {
  constructor(tag='div') { this.tag=tag; this.children=[]; this.events={}; this.dataset={}; this.value=''; this.style={};
    this.attributes={}; const classes=new Set();
    this.classList={add:c=>classes.add(c),remove:c=>classes.delete(c),contains:c=>classes.has(c),
      toggle:(c,on)=>{if(on===undefined)on=!classes.has(c);if(on)classes.add(c);else classes.delete(c);}}; }
  addEventListener(name,handler) { this.events[name]=handler; }
  append(...children) { this.children.push(...children); }
  get firstElementChild() { return this.children[0]; }
  replaceChildren(...children) { this.children=[]; this.append(...children); }
  setAttribute(name,value) {this.attributes[name]=value;} removeAttribute() {}
  scrollIntoView() {this.scrollCalls=(this.scrollCalls||0)+1;}
  reportValidity() { return Boolean(String(this.value).trim()); }
  querySelectorAll(tag) { const found=[]; const visit=e=>{for(const c of e.children||[]){
    if(c.tag===tag)found.push(c); visit(c);}}; visit(this); return found; }
}
const elements = new Map();
const document={addEventListener(){},createElement:tag=>new Element(tag),createDocumentFragment:()=>new Element(),
  getElementById:id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);},
  querySelector:()=>new Element(),querySelectorAll:()=>[],body:new Element('body')};
const context=vm.createContext({document,console,URLSearchParams,Intl,AbortController,performance,
  window:{addEventListener(){},clearTimeout(){},setTimeout(){},requestAnimationFrame:callback=>callback(),matchMedia:()=>({matches:false}),
    location:{pathname:'/',search:'',hash:''},history:{replaceState(...args){this.url=args[2];}}}});
vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
context.assert=assert;
vm.runInContext(`(async()=>{
  const realLoadItems=loadItems, realLoadTree=loadTree, realLoadDetail=loadDetail, realSelectItem=selectItem;
  collectDom(); loadItems=async()=>{}; loadDashboard=async()=>{}; loadTree=async()=>{}; showToast=()=>{};
  bindEvents();
  dom.aiStatusFilter.value='ready';dom.aiStatusFilter.events.change();
  assert.strictEqual(state.filters.ai_status,'ready');
  assert.strictEqual(state.filters.review_status,'pending');
  assert.strictEqual(state.filters.review_required,'true');
  assert.strictEqual(filterParams().get('ai_status'),'ready');
  state.filters.review_status='approved'; dom.reviewStatusFilter.value='approved';
  dom.reviewRequiredFilter.checked=true; dom.reviewRequiredFilter.events.change();
  assert.strictEqual(state.filters.review_status,'pending');
  assert.strictEqual(state.filters.review_required,'true');
  assert.strictEqual(dom.reviewStatusFilter.value,'pending');
  dom.reviewStatusFilter.value='approved'; dom.reviewStatusFilter.events.change();
  assert.strictEqual(dom.reviewRequiredFilter.checked,false);
  assert.strictEqual(state.filters.review_required,'');
  assert.strictEqual(state.filters.ai_status,'');
  assert.strictEqual(dom.aiStatusFilter.value,'');
  const optional=normalizeReview({review_required:false,review_status:'approved'});
  assert.strictEqual(optional.status,'approved'); assert.strictEqual(optional.dataset_usable,true);
  assert.strictEqual(normalizeReview({review_required:false}).dataset_usable,true);
  assert.strictEqual(normalizeReview({review_required:true}).dataset_usable,false);
  clearFilters(false); setViewMode(true,{reload:false});
  assert.strictEqual(state.filters.review_status,'pending');
  assert.strictEqual(state.filters.review_required,'true');
  assert.strictEqual(filterParams().get('has_dialogue'),'true');
  assert.strictEqual(document.body.classList.contains('annotation-mode'),true);
  assert.strictEqual(window.history.url,'/#annotate');
  dom.annotationListToggle.events.click();
  assert.strictEqual(document.body.classList.contains('annotation-list-hidden'),true);
  assert.strictEqual(dom.annotationListToggle.attributes['aria-expanded'],'false');
  dom.annotationAdvancedToggle.events.click();
  assert.strictEqual(document.body.classList.contains('annotation-advanced-open'),true);
  const raw={token:'test',uid:'test',collection:'visual_v3',category:'general',review_available:true,
    review_required:false,review_status:'not_required',content_sha256:'a'.repeat(64),
    availability:{image:false,dialogue:true,audio:false,alignment:false},
    dialogue:[{speaker:'ユーザー',text:'何が見えますか？'},{speaker:'アシスタント',text:'猫です。'}]};
  state.selectedItem=normalizeItem(raw); startDialogueEdit();
  assert.strictEqual(state.editingDialogue,true);
  assert.strictEqual(document.body.classList.contains('annotation-editing'),true);
  const inputs=dom.detailDialogue.querySelectorAll('textarea');
  assert.strictEqual(inputs.length,2); assert.strictEqual(dom.dialogueEditorFacts.querySelectorAll('input').length,0);
  assert.strictEqual(dom.dialogueHeadingActions.hidden,false);
  dom.detailDialogue.scrollTop=210;renderDialogue(state.selectedItem);
  assert.strictEqual(dom.detailDialogue.scrollTop,210);
  renderDialogue(state.selectedItem,{resetScroll:true});assert.strictEqual(dom.detailDialogue.scrollTop,0);
  // Use the newly rendered input nodes for the remaining editor assertions.
  const currentInputs=dom.detailDialogue.querySelectorAll('textarea');
  inputs.splice(0,inputs.length,...currentInputs);
  inputs[1].scrollHeight=340;inputs[1].events.input();
  assert.strictEqual(inputs[1].style.height,'342px');
  inputs[1].scrollHeight=980;inputs[1].events.input();
  assert.strictEqual(inputs[1].style.height,'982px');
  inputs[1].scrollHeight=60;inputs[1].events.input();
  assert.strictEqual(inputs[1].style.height,'96px');
  inputs[1].value='猫が写っています。';
  setViewMode(false,{reload:false});
  assert.strictEqual(filterParams().get('has_dialogue'),null);
  assert.strictEqual(state.editingDialogue,true);
  assert.strictEqual(dom.detailDialogue.querySelectorAll('textarea')[1],inputs[1]);
  setViewMode(true,{reload:false});
  assert.strictEqual(dom.detailDialogue.querySelectorAll('textarea')[1].value,'猫が写っています。');
  let saved;
  api=async(path,options)=>{assert.ok(path.endsWith('/dialogue')); saved=JSON.parse(options.body);
    return {item:{...raw,review_status:'approved',human_certified:true,dialogue:saved.dialogue},content_changed:true};};
  renderDetail=item=>{state.selectedItem=item;};
  await saveDialogueEdit();
  assert.strictEqual(dom.dialogueHeadingActions.hidden,true);
  assert.strictEqual(saved.auxiliary_task,null); assert.strictEqual(saved.dialogue[1].text,'猫が写っています。');
  assert.strictEqual(state.editingDialogue,false); assert.strictEqual(state.selectedItem.reviewState.status,'approved');
  assert.strictEqual(document.body.classList.contains('annotation-editing'),false);
  let advancedTo='';
  selectItem=async token=>{assert.strictEqual(state.savingDialogue,false);assert.strictEqual(state.reviewBusy,false);advancedTo=token;};
  state.navigation.next='next-test'; dom.annotationAutoNext.checked=true;
  startDialogueEdit(); await saveDialogueEdit();
  assert.strictEqual(advancedTo,'next-test');
  advancedTo=''; startDialogueEdit();
  api=async()=>{throw new Error('simulated conflict');}; await saveDialogueEdit();
  assert.strictEqual(advancedTo,'');assert.strictEqual(state.editingDialogue,true);
  cancelDialogueEdit();
  api=async()=>({item:{...raw,review_status:'approved'}});loadDetail=async()=>{};
  await submitReview('approved');assert.strictEqual(advancedTo,'next-test');
  // Slow queue/aggregate requests must not delay navigation after a durable save.
  const steps=[];
  selectItem=async token=>{assert.strictEqual(state.reviewBusy,false);
    assert.strictEqual(state.savingDialogue,false);steps.push('next:'+token);advancedTo=token;};
  loadItems=()=>{steps.push('list');return new Promise(()=>{});};
  loadDashboard=()=>{steps.push('dashboard');return new Promise(()=>{});};
  loadTree=()=>{steps.push('tree');return new Promise(()=>{});};
  loadDetail=()=>{throw new Error('Do not refetch the approved record before advancing');};
  api=async()=>{steps.push('saved');return {item:{...raw,review_status:'approved'}};};
  await submitReview('approved');
  assert.strictEqual(JSON.stringify(steps),JSON.stringify(['saved','next:next-test']));
  steps.length=0;startDialogueEdit();await saveDialogueEdit();
  assert.strictEqual(JSON.stringify(steps),JSON.stringify(['saved','next:next-test']));
  loadItems=async()=>{};loadDashboard=async()=>{};loadTree=async()=>{};
  // Aggregate updates are debounced and cannot overlap each other.
  let scheduled,clears=0;
  window.setTimeout=callback=>{scheduled=callback;return 42;};window.clearTimeout=()=>{clears++;};
  refreshAfterAnnotation();refreshAfterAnnotation();assert.ok(clears>=1);
  let finishDashboard,aggregateCalls=0;
  loadDashboard=()=>{aggregateCalls++;return new Promise(resolve=>{finishDashboard=resolve;});};
  const updating=scheduled();assert.strictEqual(aggregateCalls,1);
  refreshAfterAnnotation();await scheduled();assert.strictEqual(aggregateCalls,1);
  finishDashboard();await updating;loadDashboard=async()=>{aggregateCalls++;};
  await scheduled();assert.strictEqual(aggregateCalls,2);
  window.setTimeout=()=>{};window.clearTimeout=()=>{};
  loadDashboard=async()=>{};loadDetail=async()=>{};
  advancedTo='';api=async()=>{throw new Error('simulated failure');};
  await submitReview('rejected');assert.strictEqual(advancedTo,'');
  renderItems=()=>{};updatePagination=()=>{};setListState=()=>{};updateCategoryOptions=()=>{};
  api=async()=>({items:[raw],total:1,page:1});
  state.selectedToken='';await realLoadItems();assert.strictEqual(advancedTo,'test');
  advancedTo='';state.selectedToken='kept-selection';await realLoadItems();
  assert.strictEqual(advancedTo,'');
  state.editingDialogue=true;await realLoadItems({selectFirst:true});assert.strictEqual(advancedTo,'');
  state.editingDialogue=false;await realLoadItems({selectFirst:true});assert.strictEqual(advancedTo,'test');
  api=async()=>({collections:[{id:'japanese_daily_v1',label:'日常',filter:{collection:'japanese_daily_v1'},count:1}]});
  await realLoadTree();assert.strictEqual(JSON.stringify(dom.annotationCollection.children.map(option=>option.value)),JSON.stringify(['','japanese_daily_v1']));
  state.selectedItem=normalizeItem(raw);state.selectedToken='test';dom.detailContent.hidden=false;
  let deliver;api=()=>new Promise(resolve=>{deliver=resolve;});
  dom.detailPanel.scrollCalls=0;
  const refreshing=realLoadDetail('test');
  assert.strictEqual(dom.detailContent.hidden,false);assert.strictEqual(dom.detailLoading.hidden,true);
  deliver(raw);await refreshing;assert.strictEqual(dom.detailPanel.scrollCalls,0);
  api=async()=>raw;await realLoadDetail('test',{scrollToStart:true});
  assert.strictEqual(dom.detailPanel.scrollCalls,1);
  let pauses=0,plays=0;
  dom.detailAudio.pause=()=>{pauses++;};dom.detailAudio.load=()=>{};
  dom.detailAudio.play=()=>{plays++;return Promise.resolve();};
  dom.audioSection.open=false;dom.alignmentSection.open=false;
  renderAudio(normalizeItem({...raw,availability:{...raw.availability,audio:true}}));
  assert.strictEqual(dom.audioSection.open,false);
  seekAudio(1.25);assert.strictEqual(dom.audioSection.open,true);
  assert.strictEqual(dom.detailAudio.currentTime,1.25);assert.strictEqual(plays,1);
  dom.audioSection.open=false;const beforePause=pauses;dom.audioSection.events.toggle();
  assert.strictEqual(pauses,beforePause+1);
  dom.alignmentSection.open=true;
  renderAlignment({alignment:[{start:0,end:1,speaker:'assistant',text:'猫です。'}]});
  assert.strictEqual(dom.alignmentSection.open,true);
  // Prefetch the next question/image while the current item is being reviewed.
  // A successful durable PUT may then advance without another detail request.
  clearDetailPrefetch();state.excludedReviewTokens.clear();state.reviewQueueScope='';
  state.filters={collection:'',category:'',directory:'',stage:'',review_status:'pending',
    auxiliary_kind:'',review_required:'true',q:''};state.page=1;
  state.annotationMode=true;state.editingDialogue=false;state.reviewBusy=false;
  dom.annotationAutoNext.checked=true;
  loadDetail=realLoadDetail;selectItem=realSelectItem;loadItems=async()=>{};
  const raws=['a','b','c'].map(token=>({...raw,token,uid:token,review_required:true,
    review_status:'pending',content_sha256:token.repeat(64),availability:{...raw.availability,image:true}}));
  state.items=raws.map(normalizeItem);state.total=3;state.pages=1;
  state.selectedItem=normalizeItem(raws[0]);state.selectedToken='a';state.navigation={previous:'',next:'b'};
  const prefetchedRequests=[],imageRequests=[],renders=[];
  globalThis.Image=class {set src(value){imageRequests.push(value);} decode(){return Promise.resolve();}};
  renderDetail=item=>{state.selectedItem=item;state.selectedToken=item.token;renders.push(item.token);};
  let commitReview;
  api=(path,options={})=>{
    if(options.method==='PUT')return new Promise(resolve=>{commitReview=resolve;});
    prefetchedRequests.push(path);
    const token=decodeURIComponent(path.match(/items\\/([^?]+)/)[1]);
    const index=raws.findIndex(row=>row.token===token);
    return Promise.resolve({...raws[index],navigation:{previous:raws[index-1]?.token,next:raws[index+1]?.token}});
  };
  prefetchNextDetails();await Promise.all([...state.detailPrefetch.values()].map(entry=>entry.promise));
  assert.strictEqual(prefetchedRequests.length,2);
  assert.ok(imageRequests.some(url=>url.includes('/b/media/image?v='+('b'.repeat(64)))));
  const saving=submitReview('approved');
  assert.strictEqual(state.selectedToken,'a');assert.strictEqual(state.reviewBusy,true);
  assert.strictEqual(renders.length,0); // No success or navigation before durable acknowledgement.
  commitReview({item:{...raws[0],review_status:'approved'}});await saving;
  assert.strictEqual(state.selectedToken,'b');
  assert.deepStrictEqual([...renders],['b']); // Do not redraw/reload the old image/audio first.
  assert.strictEqual(prefetchedRequests.filter(path=>path.includes('/b?')).length,1);
  assert.strictEqual(state.navigation.previous,''); // Prefetched B originally pointed back to approved A.
  assert.strictEqual(state.items.some(item=>item.token==='a'),false);
  assert.strictEqual(state.total,2);
  assert.notStrictEqual(mediaUrl({...raws[1],content_sha256:'new-image-version'},'image'),mediaUrl(raws[1],'image'));
  // Selecting a prefetched record still in flight reuses the exact same request.
  clearDetailPrefetch();state.navigation.next='c';state.items=[normalizeItem(raws[1]),normalizeItem(raws[2])];
  const inFlight=[];let deliverC;
  api=path=>{inFlight.push(path);return new Promise(resolve=>{deliverC=resolve;});};
  prefetchNextDetails();const openingC=selectItem('c');
  assert.strictEqual(inFlight.length,1);
  deliverC({...raws[2],navigation:{previous:'b',next:null}});await openingC;
  assert.strictEqual(state.selectedItem.token,'c');
  // A late request which ignored AbortSignal must not replace a newer selection.
  state.annotationMode=false;clearDetailPrefetch();
  const deliveries={};api=path=>new Promise(resolve=>{deliveries[decodeURIComponent(path.match(/items\\/([^?]+)/)[1])]=resolve;});
  const oldSelection=realLoadDetail('a'),newSelection=realLoadDetail('b');
  deliveries.b(raws[1]);await newSelection;deliveries.a(raws[0]);await oldSelection;
  assert.strictEqual(state.selectedItem.token,'b');
  // Cache entries are scoped by filters; a new queue cannot consume old navigation.
  state.annotationMode=true;clearDetailPrefetch();state.navigation.next='c';state.items=[normalizeItem(raws[1]),normalizeItem(raws[2])];
  let scopedCalls=0;api=async path=>{scopedCalls++;return raws[2];};
  prefetchNextDetails();await Promise.all([...state.detailPrefetch.values()].map(entry=>entry.promise));
  state.filters.q='different-filter';await selectItem('c');assert.strictEqual(scopedCalls,2);
  // AI drafts stay pending in the UI and approval saves the text actually shown.
  state.annotationMode=false;state.navigation.next='';state.editingDialogue=false;
  const draft={...raw,review_required:true,review_status:'pending',
    dialogue:[raw.dialogue[0],{speaker:'アシスタント',text:'犬が写っています。'}],
    original_dialogue:raw.dialogue,
    auxiliary_task:{kind:'counting',answer_count:2,counted_target:'犬'},
    original_auxiliary_task:{kind:'counting',answer_count:1,counted_target:'犬'},
    ai_suggestion:{suggestion_id:'draft-1',verdict:'corrected',
      changes:[{turn_index:1,reason:'画像を再確認'}],concerns:['小さい物体に注意']}};
  state.selectedItem=normalizeItem(draft);renderReview(state.selectedItem);renderAiSuggestion(state.selectedItem);
  assert.strictEqual(state.selectedItem.reviewState.status,'pending');
  assert.strictEqual(dom.aiSuggestion.hidden,false);
  assert.strictEqual(dom.aiSuggestionConcerns.hidden,false);
  assert.ok(dom.aiSuggestionLabel.textContent.includes('未承認'));
  assert.ok(dom.reviewApprove.textContent.includes('AI修正案'));
  const diff=dom.aiSuggestionDiff.querySelectorAll('p').map(node=>node.textContent).join(' ');
  assert.ok(diff.includes('猫です。'));assert.ok(diff.includes('犬が写っています。'));
  assert.ok(diff.includes('正解の数: 1 → 2'));
  const requests=[];
  renderDetail=item=>{state.selectedItem=item;};
  api=async(path,options)=>{const body=JSON.parse(options.body);requests.push({path,body});
    return {item:{...draft,dialogue:body.dialogue,auxiliary_task:body.auxiliary_task,
      ai_suggestion:null,review_status:'approved',content_sha256:'d'.repeat(64)}};};
  await submitReview('approved');
  assert.ok(requests[0].path.endsWith('/dialogue'));
  assert.strictEqual(requests[0].body.ai_suggestion_id,'draft-1');
  assert.strictEqual(requests[0].body.content_sha256,raw.content_sha256);
  assert.strictEqual(requests[0].body.dialogue[1].text,'犬が写っています。');
  assert.strictEqual(requests[0].body.auxiliary_task.answer_count,2);
  assert.strictEqual(state.selectedItem.content_sha256,'d'.repeat(64));
  assert.strictEqual(activeAiSuggestion(state.selectedItem),null);
  // Human edits begin from the suggestion and carry the same race protection.
  state.selectedItem=normalizeItem(draft);startDialogueEdit();
  dom.detailDialogue.querySelectorAll('textarea')[1].value='犬が2匹います。';
  await saveDialogueEdit();
  assert.strictEqual(requests[1].body.ai_suggestion_id,'draft-1');
  assert.strictEqual(requests[1].body.dialogue[1].text,'犬が2匹います。');
  // An intervention after opening a draft must neither approve nor navigate.
  state.selectedItem=normalizeItem(draft);api=async()=>{throw new Error('human decision changed');};
  await submitReview('approved');
  assert.strictEqual(state.selectedItem.reviewState.status,'pending');
  assert.strictEqual(state.selectedItem.ai_suggestion.suggestion_id,'draft-1');
  assert.strictEqual(dom.reviewError.hidden,false);
  // Rejecting a draft still records a human rejection without saving AI text.
  api=async(path,options)=>{requests.push({path,body:JSON.parse(options.body)});return {decision:'rejected'};};
  await submitReview('rejected');
  assert.ok(requests[2].path.endsWith('/review'));
  assert.strictEqual(requests[2].body.ai_suggestion_id,undefined);
  assert.strictEqual(state.selectedItem.dialogue[1].text,raw.dialogue[1].text);
  assert.strictEqual(activeAiSuggestion(state.selectedItem),null);
  // Newly completed suggestions refresh a filtered queue, preserving any draft.
  state.filters.ai_status='ready';state.aiSuggestionCount=1;
  state.editingDialogue=true;let aiQueueLoads=0;
  loadItems=async()=>{aiQueueLoads++;};
  api=async()=>({ready:true,ai_suggestion_count:2});
  await loadStatus();assert.strictEqual(aiQueueLoads,0);
  assert.strictEqual(state.aiListRefreshPending,true);
  state.editingDialogue=false;await loadStatus();assert.strictEqual(aiQueueLoads,1);
  assert.strictEqual(state.aiListRefreshPending,false);
  clearFilters(false);assert.strictEqual(state.filters.ai_status,'');
  const marker={kind:'ocr',grounding_source:'vlm_visual',question_turn_index:0,answer_turn_index:1};
  const count={...marker,kind:'counting',question_turn_index:2,answer_turn_index:3};
  const dual={...marker,additional_tasks:[count,{...count,kind:'ocr'}]};
  const marked={...raw,review_required:true,review_status:'pending',dialogue:[...raw.dialogue,...raw.dialogue]};
  state.selectedItem=normalizeItem({...marked,auxiliary_task:dual});
  renderReview(state.selectedItem);startDialogueEdit();
  assert.strictEqual(dom.reviewQa.querySelectorAll('span').map(node=>node.textContent).join('|'),
    '発話1　OCR質問|発話2　OCR回答|発話3　カウント質問・OCR質問|発話4　カウント回答・OCR回答');
  assert.strictEqual(dom.detailDialogue.querySelectorAll('article')
    .filter(node=>node.classList.contains('is-review-target')).length,4);
  assert.strictEqual(dom.dialogueEditorFacts.querySelectorAll('input').length,0);
  cancelDialogueEdit();
  state.selectedItem=normalizeItem({...marked,auxiliary_task:count});startDialogueEdit();
  assert.strictEqual(dom.dialogueEditorFacts.querySelectorAll('input').length,0);
  cancelDialogueEdit();

  // The annotation entry point selects only complete, pending Terra conversations.
  state.navigation.next='';dom.annotationAutoNext.checked=false;openTerraQueue({reload:false});
  assert.strictEqual(window.history.url,'/#annotate-terra');
  assert.strictEqual(filterParams().get('ai_status'),'regenerated');
  assert.strictEqual(state.selectedItem,null);
  state.selectedItem=normalizeItem(raw);
  api=async()=>{throw new Error('A missing candidate must not send a decision');};
  await submitReview('approved');assert.strictEqual(state.reviewBusy,false);
  const info={candidate_id:'terra-v10',model:'gpt-6.1-terra',prompt_version:'v10',
    review_warnings:[{turn:2,code:'spoken_form'}]};
  const terra={...marked,regeneration_candidate:info,original_dialogue:raw.dialogue,auxiliary_task:dual};
  state.selectedItem=normalizeItem(terra);renderReview(state.selectedItem);renderAiSuggestion(state.selectedItem);
  assert.ok(dom.dialogueSectionTitle.textContent.includes('Terra'));
  assert.ok(dom.reviewApprove.textContent.includes('Terra'));
  assert.ok(dom.aiSuggestionDiff.querySelectorAll('p').some(node=>node.textContent.includes('猫です。')));
  let candidateSave;
  api=async(path,opts)=>{candidateSave={path,body:JSON.parse(opts.body)};return {item:{...terra,
    regeneration_candidate:null,review_status:'approved',human_revision:{regeneration_candidate_id:info.candidate_id}}};};
  await submitReview('approved');
  assert.ok(candidateSave.path.endsWith('/review'));
  assert.strictEqual(candidateSave.body.regeneration_candidate_id,info.candidate_id);
  assert.strictEqual(candidateSave.body.dialogue,undefined);
  assert.ok(Number.isInteger(candidateSave.body.active_ms));
  assert.strictEqual(candidateSave.body.content_sha256,raw.content_sha256);
  state.selectedItem=normalizeItem(terra);renderReview(state.selectedItem);startDialogueEdit();
  const modes=dom.dialogueEditorFacts.querySelectorAll('select');
  assert.strictEqual(modes.length,3);modes[0].value='partial';modes[1].value='lower_bound';
  await saveDialogueEdit();
  assert.ok(candidateSave.path.endsWith('/dialogue'));
  assert.strictEqual(candidateSave.body.regeneration_candidate_id,info.candidate_id);
  assert.strictEqual(candidateSave.body.dialogue.length,4);
  assert.strictEqual(candidateSave.body.auxiliary_task.answer_mode,'partial');
  assert.strictEqual(candidateSave.body.auxiliary_task.additional_tasks[0].answer_mode,'lower_bound');
  assert.strictEqual(dual.additional_tasks[0].answer_mode,undefined);
  state.selectedItem=normalizeItem(terra);await submitReview('rejected');
  assert.strictEqual(candidateSave.body.regeneration_candidate_id,info.candidate_id);
  assert.strictEqual(candidateSave.body.decision,'rejected');
  state.regenerationCandidateCount=100;state.editingDialogue=true;let arrivals=0;
  loadItems=async()=>{arrivals++;};
  api=async()=>({ready:true,regeneration_candidate_count:120});
  await loadStatus();assert.strictEqual(arrivals,0);
  state.editingDialogue=false;await loadStatus();assert.strictEqual(arrivals,1);
  // The first arrival after an empty queue opens its conversation without a click.
  openTerraQueue({reload:false});loadItems=realLoadItems;let firstOpened='',queue=[];
  selectItem=async token=>{firstOpened=token;state.selectedToken=token;};
  api=async path=>path==='/api/status'?{ready:true,regeneration_candidate_count:121}:{items:queue,total:queue.length};
  await loadItems();assert.strictEqual(firstOpened,'');
  queue=[terra];await loadStatus();assert.strictEqual(firstOpened,terra.token);
})()`,context).catch(error=>{console.error(error);process.exitCode=1;});
"""
        result = subprocess.run([shutil.which("node"), "-e", script, str(source)],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
