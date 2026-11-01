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
  reportValidity() { return Boolean(this.value.trim()); }
  querySelectorAll(tag) { const found=[]; const visit=e=>{for(const c of e.children||[]){
    if(c.tag===tag)found.push(c); visit(c);}}; visit(this); return found; }
}
const elements = new Map();
const document={addEventListener(){},createElement:tag=>new Element(tag),createDocumentFragment:()=>new Element(),
  getElementById:id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);},
  querySelector:()=>new Element(),querySelectorAll:()=>[],body:new Element('body')};
const context=vm.createContext({document,console,URLSearchParams,Intl,AbortController,
  window:{addEventListener(){},clearTimeout(){},setTimeout(){},requestAnimationFrame:callback=>callback(),matchMedia:()=>({matches:false}),
    location:{pathname:'/',search:'',hash:''},history:{replaceState(...args){this.url=args[2];}}}});
vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
context.assert=assert;
vm.runInContext(`(async()=>{
  const realLoadItems=loadItems, realLoadTree=loadTree, realLoadDetail=loadDetail;
  collectDom(); loadItems=async()=>{}; loadDashboard=async()=>{}; loadTree=async()=>{}; showToast=()=>{};
  bindEvents();
  state.filters.review_status='approved'; dom.reviewStatusFilter.value='approved';
  dom.reviewRequiredFilter.checked=true; dom.reviewRequiredFilter.events.change();
  assert.strictEqual(state.filters.review_status,'pending');
  assert.strictEqual(state.filters.review_required,'true');
  assert.strictEqual(dom.reviewStatusFilter.value,'pending');
  dom.reviewStatusFilter.value='approved'; dom.reviewStatusFilter.events.change();
  assert.strictEqual(dom.reviewRequiredFilter.checked,false);
  assert.strictEqual(state.filters.review_required,'');
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
  assert.strictEqual(JSON.stringify(steps),JSON.stringify(['saved','next:next-test','list']));
  steps.length=0;startDialogueEdit();await saveDialogueEdit();
  assert.strictEqual(JSON.stringify(steps),JSON.stringify(['saved','next:next-test','list']));
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
})()`,context).catch(error=>{console.error(error);process.exitCode=1;});
"""
        result = subprocess.run([shutil.which("node"), "-e", script, str(source)],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
