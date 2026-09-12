from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'dataset'
out = pd.read_csv(ROOT / 'output.csv')
req = pd.read_csv(D / 'requests.csv')
opts = pd.read_csv(D / 'request_payment_options.csv')
profiles = pd.read_csv(D / 'financial_profiles.csv')
events = pd.read_csv(D / 'financial_events.csv')

errors=[]
REQCOLS=['request_id','amount_safe_to_pay','affordability_status','recommended_payment_method','payment_plan','earliest_date_for_full_payment','spending_changes_needed','decision_explanation']
for c in REQCOLS:
    if c not in out.columns: errors.append(f'Missing output column: {c}')

if len(out)!=len(req): errors.append(f'Row count mismatch: output={len(out)} requests={len(req)}')

reqmap=req.set_index('request_id').to_dict('index')
opt_groups={k:g.copy() for k,g in opts.groupby('request_id')}
prof=profiles.set_index('user_id').to_dict('index')
ev=events.set_index('event_id')

for r in out.itertuples(index=False):
    rid=r.request_id
    if rid not in reqmap: errors.append(f'{rid}: not in requests.csv'); continue
    q=reqmap[rid]; user=prof[q['user_id']]
    amount=float(q['requested_amount']); safe=float(r.amount_safe_to_pay)
    status=str(r.affordability_status); method=str(r.recommended_payment_method)
    plan='' if pd.isna(r.payment_plan) else str(r.payment_plan)
    earliest='' if pd.isna(r.earliest_date_for_full_payment) else str(r.earliest_date_for_full_payment)
    changes='' if pd.isna(r.spending_changes_needed) else str(r.spending_changes_needed)

    if safe < -1e-9 or safe > amount + 1e-9: errors.append(f'{rid}: safe amount out of bounds {safe}>{amount}')
    if status=='not_affordable' and method!='not_recommended': errors.append(f'{rid}: not_affordable method={method}')
    if status=='not_affordable' and plan.lower()!='none': errors.append(f'{rid}: not_affordable plan={plan}')
    if status!='not_affordable' and earliest.strip()=='': errors.append(f'{rid}: missing earliest date')
    if status=='affordable_now' and method!='full_payment': errors.append(f'{rid}: affordable_now method={method}')
    if status=='affordable_now' and earliest[:10] != str(q['request_date'])[:10]: errors.append(f'{rid}: affordable_now earliest={earliest} request_date={q["request_date"]}')

    parts=[] if plan.lower() in ('none','nan','') else plan.split('|')
    dates=[]; vals=[]
    for p in parts:
        if ':' not in p: errors.append(f'{rid}: malformed plan component {p}'); continue
        ds,vs=p.split(':',1)
        try: dates.append(pd.Timestamp(ds)); vals.append(float(vs))
        except Exception: errors.append(f'{rid}: malformed plan component {p}')
    if dates and dates != sorted(dates): errors.append(f'{rid}: payment dates not sorted')

    if method=='full_payment':
        if len(parts)!=1: errors.append(f'{rid}: full_payment must have one payment, got {len(parts)}')
        elif abs(vals[0]-amount)>0.02: errors.append(f'{rid}: full payment amount {vals[0]} != requested {amount}')

    if method=='partial_payment':
        if not bool(q['allows_partial_payment']): errors.append(f'{rid}: partial_payment when request disallows it')
        if len(parts)!=2: errors.append(f'{rid}: partial_payment requires exactly 2 payments, got {len(parts)}')
        elif abs(dates[0]-pd.Timestamp(q['request_date']))>pd.Timedelta(seconds=1): errors.append(f'{rid}: first partial payment not on request date')
        elif abs(vals[0]-safe)>0.02 or abs((vals[0]+vals[1])-amount)>0.02: errors.append(f'{rid}: partial amounts inconsistent safe={safe} vals={vals} requested={amount}')
        if len(parts)==2 and q['desired_completion_date'] and dates[1] > pd.Timestamp(q['desired_completion_date']): errors.append(f'{rid}: partial second payment after desired completion')

    if method=='installments':
        matching=[]
        for _,o in opt_groups.get(rid,pd.DataFrame()).iterrows():
            if str(o['payment_method'])!='installments': continue
            if pd.isna(o['number_of_payments']): continue
            n=int(o['number_of_payments']); first=pd.Timestamp(o['first_payment_date']); freq=int(o['payment_frequency_days']) if not pd.isna(o['payment_frequency_days']) else 0; pa=float(o['payment_amount']);
            od=[first+pd.Timedelta(days=freq*i) for i in range(n)]
            ov=[pa]*n
            if len(parts)==n and all(abs((dates[i]-od[i]).total_seconds())<1 for i in range(n)) and all(abs(vals[i]-ov[i])<0.02 for i in range(n)):
                matching.append(o)
        if not matching: errors.append(f'{rid}: installment plan does not exactly match any supplied installment option')
        maxm=user.get('max_installment_months')
        if matching and not pd.isna(maxm):
            n=int(matching[0]['number_of_payments'])
            if n>int(maxm): errors.append(f'{rid}: installment count {n} exceeds max {int(maxm)}')
        if parts and q['desired_completion_date'] and dates[-1] > pd.Timestamp(q['desired_completion_date']): errors.append(f'{rid}: installment completion after desired date')
        accepted=str(user['payment_methods_user_will_consider']).split('|')
        if 'installments' not in accepted: errors.append(f'{rid}: installments not accepted by user profile')

    # Spending changes: max 3 and syntactically + semantically permitted.
    if changes.strip().lower()!='none':
        cps=[x for x in changes.split('|') if x.strip()]
        if len(cps)>3: errors.append(f'{rid}: more than 3 spending changes')
        protected=set(str(user['expense_categories_to_protect']).split('|'))
        reduc=set(str(user['expense_categories_user_is_willing_to_reduce']).split('|'))
        stop=set(str(user['expense_categories_user_is_willing_to_stop']).split('|'))
        for c in cps:
            if c.startswith('stop:'):
                eid=c[5:]
                if eid not in ev.index: errors.append(f'{rid}: unknown stop event {eid}'); continue
                e=ev.loc[eid]
                if e['user_id']!=q['user_id']: errors.append(f'{rid}: stop event belongs to wrong user {eid}')
                if str(e['flexibility']) not in ('stoppable','reducible_or_stoppable'): errors.append(f'{rid}: event not stoppable {eid}')
                if str(e['category']) not in stop: errors.append(f'{rid}: category not permitted to stop {e["category"]} for {eid}')
                if str(e['category']) in protected: errors.append(f'{rid}: protected category stopped {eid}')
            elif c.startswith('reduce_to:'):
                bits=c.split(':',2)
                if len(bits)!=3: errors.append(f'{rid}: malformed reduce change {c}'); continue
                eid,new=bits[1],bits[2]
                if eid not in ev.index: errors.append(f'{rid}: unknown reduce event {eid}'); continue
                e=ev.loc[eid]
                if e['user_id']!=q['user_id']: errors.append(f'{rid}: reduce event belongs to wrong user {eid}')
                if str(e['flexibility']) not in ('reducible','reducible_or_stoppable'): errors.append(f'{rid}: event not reducible {eid}')
                if str(e['category']) not in reduc: errors.append(f'{rid}: category not permitted to reduce {e["category"]} for {eid}')
                if str(e['category']) in protected: errors.append(f'{rid}: protected category reduced {eid}')
                try: nv=float(new); old=float(e['amount']); minv=float(e['minimum_allowed_amount']) if not pd.isna(e['minimum_allowed_amount']) else 0
                except Exception: errors.append(f'{rid}: bad reduce amount {c}'); continue
                if not (nv>=minv-1e-9 and nv<=old+1e-9): errors.append(f'{rid}: invalid reduce amount {c} old={old} min={minv}')
            else: errors.append(f'{rid}: unknown spending action {c}')

    # Any plan must complete by desired date, unless it is a wait plan (then it may be later).
    if method not in ('wait','not_recommended') and dates and q['desired_completion_date'] and dates[-1] > pd.Timestamp(q['desired_completion_date']):
        errors.append(f'{rid}: non-wait plan ends after desired completion')

print(f'ROWS: {len(out)}')
print(f'ERROR COUNT: {len(errors)}')
if errors:
    print('\n'.join(errors[:200]))
else:
    print('ALL CHALLENGE-RULE VALIDATION CHECKS PASSED')
raise SystemExit(1 if errors else 0)
